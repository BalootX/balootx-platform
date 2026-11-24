from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ortools.sat.python import cp_model

from baloot_platform.research.ortools_proof import (
    OrToolsScenarioGraphProofV1,
    ProofGraphEdgeV1,
    ProofGraphNodeV1,
    ProofGraphV1,
    ProofGraphForcedSequenceV1,
    ProofGraphSequencePackV1,
    required_line_count_limbs_v1,
)
from baloot_platform.research.scenario_contracts import (
    ScenarioProofRequestV1,
    canonical_sha256,
)


def _edge(action_id: int, code: str, child: str) -> ProofGraphEdgeV1:
    return ProofGraphEdgeV1.build(action_id, code, child)


def test_forced_sequence_preserves_actions_observations_and_identity():
    from dataclasses import replace
    nodes = (
        ProofGraphNodeV1.decision("pass", "north_south", (_edge(0, "pass", "play"),), actor_seat="north"),
        ProofGraphNodeV1.decision("play", "north_south", (ProofGraphEdgeV1.build(
            0, "S7", "end", play_trick_index=0, completed_trick_index=0,
            completed_trick_winner_seat="north", completed_trick_winning_card_id=0),), actor_seat="north"))
    sequence = ProofGraphForcedSequenceV1.build(nodes, observation_ids={"pass": "a", "play": "b"})
    sequence.validate()
    assert sequence.to_dict()["nodes"] == [node.to_dict() for node in nodes]
    assert sequence.to_dict()["actionDepth"] == 2
    assert sequence == ProofGraphForcedSequenceV1.build(nodes, observation_ids={"play": "b", "pass": "a"})
    with pytest.raises(ValueError, match="checksum"):
        replace(sequence, sha256="corrupt").validate()
    with pytest.raises(ValueError, match="observations"):
        ProofGraphForcedSequenceV1.build(nodes, observation_ids={"pass": "a"})
    with pytest.raises(ValueError, match="connected"):
        ProofGraphForcedSequenceV1.build(tuple(reversed(nodes)))
    branching = replace(nodes[0], edges=(*nodes[0].edges, _edge(1, "declare", "end")))
    with pytest.raises(ValueError, match="single-action"):
        ProofGraphForcedSequenceV1.build((branching,))
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            _edge(0, "start", "pass"), _edge(1, "other", "end")), actor_seat="north"),
        *nodes, ProofGraphNodeV1.leaf("end", 1)))
    observations = {"root": "r", "pass": "a", "play": "b"}
    packed = ProofGraphSequencePackV1.build(graph, observation_ids=observations)
    assert len(packed.graph.nodes) == 2
    assert packed.expand() == (graph, observations)
    assert packed == ProofGraphSequencePackV1.build(graph, observation_ids=dict(reversed(list(observations.items()))))
    with pytest.raises(ValueError, match="checksum"):
        replace(packed, source_graph_hash="wrong").expand()
    service = OrToolsScenarioGraphProofV1()
    request = _request(selected_action="start", line_universe="all_legal")
    assert service.prove(packed, request) == service.prove(graph, request)
    with pytest.raises(ValueError, match="checksum"):
        service.prove(replace(packed, source_graph_hash="wrong"), request)
    winning_graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            _edge(0, "start", "pass"), _edge(1, "other", "loss")), actor_seat="north"),
        *nodes, ProofGraphNodeV1.leaf("end", 3), ProofGraphNodeV1.leaf("loss", 0)))
    winning_pack = ProofGraphSequencePackV1.build(winning_graph, observation_ids=observations)
    proof = service.prove(winning_pack, request)
    assert proof == service.prove(winning_graph, request)
    assert proof.claim_satisfied
    assert proof.winning_root_action_count == 1
    assert proof.principal_variation == ("start", "pass", "S7")
    assert proof.continuation_lines.exact_count == "1"


@pytest.mark.parametrize("method", ["_solve", "_solve_count"])
def test_graph_solves_preserve_live_cancellation(monkeypatch, method):
    import baloot_platform.research.ortools_proof as proof_module
    stopped = []
    def interrupted(solver, model, cancel_check):
        assert not cancel_check()
        stopped.append(True)
        return cp_model.OPTIMAL
    monkeypatch.setattr(proof_module, "_solve_cp_sat_with_cancel_v1", interrupted)
    with pytest.raises(InterruptedError, match="cancelled"):
        getattr(OrToolsScenarioGraphProofV1, method)(cp_model.CpModel(),
            maximum_seconds=1, cancel_check=lambda: bool(stopped))
    assert stopped


def test_line_count_limb_estimator_uses_the_proof_base() -> None:
    assert required_line_count_limbs_v1(0) == 1
    assert required_line_count_limbs_v1(999_999) == 1
    assert required_line_count_limbs_v1(1_000_000) == 2
    with pytest.raises(ValueError):
        required_line_count_limbs_v1(-1)


@pytest.mark.real_clock
def test_value_and_count_solves_share_one_deadline(monkeypatch):
    import baloot_platform.research.ortools_proof as proof_module
    elapsed = [0.0]
    monkeypatch.setattr(proof_module.time, "perf_counter", lambda: elapsed[0])
    original = OrToolsScenarioGraphProofV1._solve
    limits = []
    def solve(model, *, maximum_seconds):
        limits.append(maximum_seconds)
        result = original(model, maximum_seconds=maximum_seconds)
        elapsed[0] += 2
        return result
    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "_solve", staticmethod(solve))
    result = OrToolsScenarioGraphProofV1().prove(_graph(),
        _request(line_universe="all_legal"), maximum_seconds=1)
    assert limits == [1, -1]
    assert result.continuation_lines.complete is False
    assert result.continuation_lines.exact_count is None
    assert result.continuation_lines.lower_bound == "0"
    assert int(result.continuation_lines.upper_bound) > 0


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -float("inf"), True, 0, -1, "1"])
def test_graph_proof_refuses_invalid_time_before_solving(monkeypatch, seconds):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid budget must not reach solver")
    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "_solve", staticmethod(forbidden))
    with pytest.raises(ValueError, match="time bound"):
        OrToolsScenarioGraphProofV1().prove(_graph(), _request(), maximum_seconds=seconds)


def _graph(*, reverse: bool = False, scope: str = "exact") -> ProofGraphV1:
    nodes = [
        ProofGraphNodeV1.decision(
            "root",
            "north_south",
            [_edge(20, "B", "north"), _edge(10, "A", "west")],
        ),
        ProofGraphNodeV1.decision(
            "west",
            "west_east",
            [_edge(9, "W9", "five"), _edge(4, "W4", "one")],
        ),
        ProofGraphNodeV1.decision(
            "north",
            "north_south",
            [_edge(3, "N3", "three-a"), _edge(2, "N2", "three-b")],
        ),
        ProofGraphNodeV1.leaf("five", 5),
        ProofGraphNodeV1.leaf("one", 1),
        ProofGraphNodeV1.leaf("three-a", 3),
        ProofGraphNodeV1.leaf("three-b", 3),
    ]
    if scope == "bounded":
        nodes[-1] = ProofGraphNodeV1.leaf("three-b", 3, kind="boundary")
    if reverse:
        nodes.reverse()
    return ProofGraphV1.build(
        scope=scope,
        root_node_id="root",
        nodes=nodes,
        boundary_evaluator=(
            "current_raw_point_margin_v1" if scope == "bounded" else None
        ),
    )


def _request(
    *,
    claim: str = "selected_action_is_unique_winner",
    selected_action: str | None = "B",
    mode: str = "exact",
    line_universe: str | None = None,
    policies: dict | None = None,
) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "complete_state",
        "claim": claim,
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": 2,
        "budgetProfile": "test",
    }
    if selected_action is not None:
        value["selectedAction"] = selected_action
    if line_universe is not None:
        value["lineUniverse"] = line_universe
    if mode == "bounded":
        value["horizon"] = {"unit": "plies", "value": 2}
        value["boundaryEvaluator"] = "current_raw_point_margin_v1"
    if mode == "policy_conditioned":
        value["policies"] = policies
    return ScenarioProofRequestV1.build(value)


@pytest.mark.parametrize("scope", [
    {"mode": "bounded"}, {"mode": "policy_conditioned"},
    {"visibility": "actor_information_set"}, {"utility": "round_win"},
])
@pytest.mark.parametrize("entrypoint", ["configure", "execute", "tail"])
def test_native_delayed_refuses_unsupported_scope_before_binding(rules, scope, entrypoint):
    from dataclasses import replace
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import (
        OrToolsUnifiedResearchExecutorV1, _native_proof_tactic_configurer_v3,
    )
    from baloot_platform.research.ortools_tail_first_candidate import OrToolsTailFirstCandidateCompilerV1
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
    from baloot_platform.tactics import TacticalDetectorProfileV1

    raw_profile = TacticalDetectorProfileV1.defaults().to_dict()
    raw_profile.pop("sha256")
    profile = TacticalDetectorProfileV1.build({**raw_profile, "nativeDelayedPayoff": {
        "gainMetric": "raw_card_points", "minimumGain": 8}})
    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-DELAYED", "kind": "tactic.verified", "version": 1,
            "hardness": "hard", "arguments": {"kind": "small_card_big_gain", "causalRequired": True,
                "minimumScore": 100, "minimumEvidence": 1}},
        "decisionPoint": {"kind": "play", "selector": {}},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "state.phase", "version": 1,
            "hardness": "report", "arguments": {"state": "playing"}}})
    with pytest.raises(ValueError, match="native delayed payoff requires exact complete-state"):
        request = replace(_request(), **scope)
        if entrypoint == "configure":
            _native_proof_tactic_configurer_v3(scenario=scenario, candidate=SimpleNamespace(),
                graph=_graph(), request=request, rules=rules,
                manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(native_delayed_payoff=True),
                tactical_detector_profile=profile)
        elif entrypoint == "execute":
            executor = object.__new__(OrToolsUnifiedResearchExecutorV1)
            executor.backend_id = "ortools_cp_sat_v3"
            executor.tactical_detector_profile = profile
            executor.execute(scenario, request)
        else:
            compiler = object.__new__(OrToolsTailFirstCandidateCompilerV1)
            compiler.maximum_conflicts = 100
            compiler.proof_request = request
            compiler.tactical_detector_profile = profile
            compiler.native_cp_sat_scoring = True
            compiler(scenario, (), seed=0, maximum_seconds=1, maximum_conflicts=100)


@pytest.mark.parametrize("mode", ["exact", "bounded"])
@pytest.mark.parametrize("team", ["north_south", "west_east"])
@pytest.mark.parametrize("recorded", [None, "A", "B"])
def test_complete_state_native_quality_certifies_all_root_values(mode, team, recorded):
    graph = ProofGraphV1.build(scope=mode, root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", team, (
            _edge(20, "A", "one"), _edge(10, "B", "three"), _edge(5, "C", "three")),
            actor_seat="north" if team == "north_south" else "west"),
        ProofGraphNodeV1.leaf("one", 1), ProofGraphNodeV1.leaf("three", 3),
    ), boundary_evaluator="current_raw_point_margin_v1" if mode == "bounded" else None)
    request = _request(mode=mode, claim="winning_root_action_count", selected_action=recorded)
    service = OrToolsScenarioGraphProofV1()
    legacy = service.prove(graph, request)
    result = service.prove(graph, request, native_action_quality=True)
    model = result.model_document()
    quality = model["actionQuality"]
    best = 3 if team == "north_south" else 1
    expected_action = recorded or ("C" if team == "north_south" else "A")
    actual = 1 if expected_action == "A" else 3
    assert result.root_actions == legacy.root_actions
    assert result.completion_status == legacy.completion_status
    assert result.selected_action == expected_action
    assert quality["scope"] == "compiled_complete_state"
    assert quality["actualAction"] == expected_action
    assert quality["bestValue"] == best and quality["actualValue"] == actual
    assert quality["regret"] == abs(best - actual)
    assert quality["actualActionOptimal"] is (best == actual)
    assert set(quality["optimalActions"]) == ({"B", "C"} if team == "north_south" else {"A"})
    assert quality["sourceQualificationModelHash"] == legacy.model_hash
    assert quality["sha256"] == canonical_sha256({
        **{key: value for key, value in quality.items() if key != "sha256"},
        "modelProto": model["modelProto"],
    })
    assert "actionQuality" not in legacy.model_document()
    assert service.prove(graph, request, native_action_quality=True) == result


def test_complete_state_conflict_limit_reaches_value_and_quality_solves(monkeypatch):
    from baloot_platform.research import ortools_proof as proof_module
    from baloot_platform.research import ortools_information_set_compiler as compiler

    original = proof_module._solve_cp_sat_with_cancel_v1
    limits = []
    def observed(solver, model, cancel_check):
        limits.append(solver.parameters.max_number_of_conflicts)
        return original(solver, model, cancel_check)
    monkeypatch.setattr(proof_module, "_solve_cp_sat_with_cancel_v1", observed)
    monkeypatch.setattr(compiler, "_solve_cp_sat_with_cancel_v1", observed)
    service = OrToolsScenarioGraphProofV1()
    result = service.prove(_graph(), _request(), maximum_conflicts=31, native_action_quality=True)
    assert result.completion_status == "exact_complete"
    assert limits == [31, 31]
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="conflict limit"):
            service.prove(_graph(), _request(), maximum_conflicts=invalid)
    with pytest.raises(ValueError, match="continuation counting"):
        service.prove(_graph(), _request(line_universe="all_legal"), maximum_conflicts=31)


def test_complete_state_quality_exhaustion_never_returns_exact(monkeypatch):
    from baloot_platform.research import ortools_information_set_compiler as compiler

    def exhausted(**kwargs):
        raise compiler.InformationSetCompilationStopped("maximum_seconds", stage="action_quality")

    monkeypatch.setattr(compiler, "_certify_action_quality_model_v1", exhausted)
    result = OrToolsScenarioGraphProofV1().prove(_graph(), _request(), native_action_quality=True)
    assert result.completion_status == "proof_incomplete"
    assert result.selected_action is None and result.root_actions == ()
    assert "actionQuality" not in result.model_document()


def _native_tactic_graph(*, winner: str) -> ProofGraphV1:
    return ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                "north_south",
                (_edge(0, "S7", "selected"), _edge(8, "H7", "other")),
                actor_seat="north",
            ),
            ProofGraphNodeV1.decision(
                "selected",
                "west_east",
                (
                    ProofGraphEdgeV1.build(
                        1,
                        "S8",
                        "win",
                        play_trick_index=6,
                        completed_trick_index=6,
                        completed_trick_winner_seat=winner,
                        completed_trick_winning_card_id=1,
                    ),
                ),
                actor_seat="west",
            ),
            ProofGraphNodeV1.decision(
                "other",
                "west_east",
                (
                    ProofGraphEdgeV1.build(
                        9,
                        "H8",
                        "lose",
                        play_trick_index=6,
                        completed_trick_index=6,
                        completed_trick_winner_seat="west",
                        completed_trick_winning_card_id=9,
                    ),
                ),
                actor_seat="west",
            ),
            ProofGraphNodeV1.leaf("win", 3),
            ProofGraphNodeV1.leaf("lose", -1),
        ),
        tactic_context={
            "baseTricks": [3, 2],
            "trumpSuit": None,
            "ownerByCard": [0] * 32,
        },
    )


@pytest.mark.parametrize("trump_suit,applicable,invalid", [
    (None, False, None), (0, True, None), (1, False, None),
    (None, False, "missing_key"), (None, False, "missing_hokum"),
    (0, True, "missing_profile"),
])
@pytest.mark.parametrize("selected_action", ["S7", None])
def test_native_trump_control_binds_no_trump_as_non_applicable(rules, trump_suit, applicable, invalid, selected_action):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    original = _native_tactic_graph(winner="north")
    context = {**original.tactic_context, "trumpSuit": trump_suit}
    if invalid == "missing_key":
        context.pop("trumpSuit")
    graph = ProofGraphV1.build(scope=original.scope, root_node_id=original.root_node_id,
        nodes=original.nodes, tactic_context=context)
    request = _request(selected_action=selected_action, claim="winning_root_action_count")
    for requested in (False, True):
        scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
            "constraints": {"predicateId": "PRED-TRUMP", "kind": "tactic.applicable",
                "version": 1, "hardness": "hard",
                "arguments": {"kind": "trump_control", "applicable": requested}},
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round",
                "version": 1, "hardness": "report", "arguments": {"team": "north_south", "status": "won"}}})
        def certify():
            configure = _native_proof_tactic_configurer_v3(scenario=scenario,
                candidate=SimpleNamespace(trace={}, state={"contract": "sun" if trump_suit is None and invalid != "missing_hokum" else "hokum",
                    "trumpSuit": trump_suit}), graph=graph, request=request, rules=rules,
                manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
                tactical_detector_profile=None if invalid == "missing_profile" else TacticalDetectorProfileV1.defaults())
            return OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
        if invalid:
            with pytest.raises(ValueError, match="bindings are unavailable" if invalid == "missing_key"
                               else "frozen tactical detector profile" if invalid == "missing_profile" else "solved contract"):
                certify()
            continue
        proof = certify()
        assert proof.completion_status == "exact_complete"
        evidence = proof.native_qualification["PRED-TRUMP"]
        assert evidence["satisfied"] is (requested == applicable)
        assert evidence["observed"]["applicable"] is applicable
        assert evidence["observed"]["selectedRootAction"] == proof.selected_action


@pytest.mark.parametrize('hardness', ['hard', 'report', 'prefer'])
@pytest.mark.parametrize('selected_action', ['S9', None])
@pytest.mark.parametrize('minimum_gain', [0, 4, 4.001])
def test_native_trump_extraction_uses_only_selected_post_decision_cards(rules, hardness, selected_action, minimum_gain):
    from dataclasses import replace
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    original = _native_tactic_graph(winner='north')
    profile = TacticalDetectorProfileV1.defaults().to_dict()
    profile.pop('sha256')
    profile = TacticalDetectorProfileV1.build({**profile, 'minimumCausalGain': minimum_gain})
    request = _request(selected_action=selected_action, claim='winning_root_action_count')
    for draws, prefix_only in ((True, False), (False, False), (False, True)):
        nodes = tuple(replace(node, edges=tuple(
            replace(edge, action_id=2, action_code='S9', play_trick_index=6)
            if node.node_id == 'root' and edge.action_code == 'S7' else
            replace(edge, action_id=1 if draws else 9, action_code='S8' if draws else 'H8',
                completed_trick_winning_card_id=2)
            if node.node_id == 'selected' else edge for edge in node.edges)) for node in original.nodes)
        graph = ProofGraphV1.build(scope=original.scope, root_node_id=original.root_node_id,
            nodes=nodes, tactic_context={**original.tactic_context, 'trumpSuit': 0})
        predicates = [{'predicateId': 'PRED-EXTRACTION', 'kind': 'tactic.verified', 'version': 1,
            'hardness': hardness, 'arguments': {'kind': 'trump_control', 'minimumScore': 100,
                'minimumEvidence': 1, 'causalRequired': True}}]
        if prefix_only:
            predicates.append({'predicateId': 'PRED-PREFIX-COUNT',
                'kind': 'proof.principal_variation_trick_team_rank_count', 'version': 1, 'hardness': 'hard',
                'arguments': {'trickIndex': 6, 'team': 'west_east', 'rank': ['ten'], 'minimum': 1, 'maximum': 1}})
        scenario = ResearchScenarioV1.build({'schemaVersion': 1, 'origin': {'kind': 'search'},
            'constraints': {'op': 'all', 'children': predicates},
            'decisionPoint': {'kind': 'play', 'selector': {}},
            'outcome': {'predicateId': 'PRED-REPORT', 'kind': 'outcome.team_round',
                'version': 1, 'hardness': 'report', 'arguments': {'team': 'north_south', 'status': 'won'}}})
        configure = _native_proof_tactic_configurer_v3(scenario=scenario,
            candidate=SimpleNamespace(trace={}, state={'contract': 'hokum', 'trumpSuit': 0,
                'currentTrick': {'index': 6, 'plays': [{'actor': 'east', 'card': 'S10'}]} if prefix_only else None}),
            graph=graph, request=request, rules=rules,
            manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(), tactical_detector_profile=profile)
        proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
        assert proof.completion_status == 'exact_complete'
        evidence = proof.native_qualification['PRED-EXTRACTION']
        assert evidence['satisfied'] is (draws and minimum_gain <= 4)
        assert evidence['observed']['trumpExtraction'] is draws
        assert evidence['observed']['causalProven'] is (draws and minimum_gain <= 4)
        assert evidence['observed']['mechanismKind'] == 'trump_extraction'
        assert evidence['observed']['mechanismScope'] == 'selected_canonical_continuation'
        assert evidence['observed']['detectorVersion'] == '3.0.1'
        assert evidence['observed']['detectorProfileHash'] == profile.sha256
        assert evidence['observed']['minimumCausalGain'] == minimum_gain
        assert evidence['observed']['causalGainUnit'] == 'awarded_point_margin'
        assert evidence['observed']['causalGainScale'] == 1
        assert evidence['observed']['minimumCausalGainScaled'] == (1 if minimum_gain == 0 else 4 if minimum_gain == 4 else 5)
        if prefix_only:
            assert proof.native_qualification['PRED-PREFIX-COUNT']['satisfied'] is True


@pytest.mark.parametrize('target', ['north_south', 'west_east'])
@pytest.mark.parametrize('scale,minimum_gain,gap,expected', [
    (1000, 4, 4, True), (1000, 4.001, 4, False),
    (10, 4.001, 4, False), (1000, 0, 0, False),
])
def test_native_trump_gain_uses_actor_relative_scaled_awarded_points(rules, target, scale, minimum_gain, gap, expected):
    """Finite model boundary check, not a reachable-deal qualification."""
    from dataclasses import replace
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    original = _native_tactic_graph(winner='north')
    direction = 1 if target == 'north_south' else -1
    nodes = tuple(replace(node, utility=(gap - 1 if node.node_id == 'win' else -1) * scale * direction)
        if node.is_leaf else replace(node, edges=tuple(
            replace(edge, action_id=2, action_code='S9', play_trick_index=6)
            if node.node_id == 'root' and edge.action_code == 'S7' else
            replace(edge, completed_trick_winning_card_id=2) if node.node_id == 'selected' else edge
            for edge in node.edges)) for node in original.nodes)
    graph = ProofGraphV1.build(scope='exact', root_node_id='root', nodes=nodes,
        utility_scale=scale, tactic_context={**original.tactic_context, 'trumpSuit': 0})
    scenario = ResearchScenarioV1.build({'schemaVersion': 1, 'origin': {'kind': 'search'},
        'constraints': {'predicateId': 'PRED-TRUMP', 'kind': 'tactic.verified', 'version': 1,
            'hardness': 'hard', 'arguments': {'kind': 'trump_control', 'minimumScore': 100,
                'minimumEvidence': 1, 'causalRequired': True}},
        'decisionPoint': {'kind': 'play', 'selector': {}},
        'outcome': {'predicateId': 'PRED-COUNT', 'kind': 'proof.winning_root_action_count',
            'version': 1, 'hardness': 'report', 'arguments': {'minimum': 0, 'maximum': 2}}})
    raw_request = _request(selected_action='S9').to_dict()
    raw_request.pop('sha256')
    request = ScenarioProofRequestV1.build({**raw_request, 'targetTeam': target})
    raw_profile = TacticalDetectorProfileV1.defaults().to_dict()
    raw_profile.pop('sha256')
    profile = TacticalDetectorProfileV1.build({**raw_profile, 'minimumCausalGain': minimum_gain})
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={'contract': 'hokum', 'trumpSuit': 0}),
        graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(), tactical_detector_profile=profile)
    proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
    evidence = proof.native_qualification['PRED-TRUMP']
    assert evidence['satisfied'] is expected
    assert evidence['observed']['causalGainScale'] == scale
    assert evidence['observed']['minimumCausalGain'] == minimum_gain


@pytest.mark.parametrize('mode,utility', [('bounded', 'awarded_point_margin'), ('exact', 'round_win')])
def test_native_trump_control_refuses_nonterminal_awarded_scope(rules, mode, utility):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    original = _native_tactic_graph(winner='north')
    graph = ProofGraphV1.build(scope=mode, root_node_id=original.root_node_id,
        nodes=original.nodes, tactic_context={**original.tactic_context, 'trumpSuit': 0},
        boundary_evaluator='current_raw_point_margin_v1' if mode == 'bounded' else None)
    scenario = ResearchScenarioV1.build({'schemaVersion': 1, 'origin': {'kind': 'search'},
        'constraints': {'predicateId': 'PRED-TRUMP', 'kind': 'tactic.verified', 'version': 1,
            'hardness': 'hard', 'arguments': {'kind': 'trump_control', 'minimumScore': 100, 'minimumEvidence': 1}},
        'decisionPoint': {'kind': 'play', 'selector': {}},
        'outcome': {'predicateId': 'PRED-COUNT', 'kind': 'proof.winning_root_action_count',
            'version': 1, 'hardness': 'report', 'arguments': {'minimum': 0, 'maximum': 2}}})
    raw = _request(selected_action='S7', mode=mode).to_dict()
    raw.pop('sha256')
    request = ScenarioProofRequestV1.build({**raw, 'utility': utility})
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={'contract': 'hokum', 'trumpSuit': 0}),
        graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
        tactical_detector_profile=TacticalDetectorProfileV1.defaults())
    with pytest.raises(ValueError, match='terminal awarded-point proof'):
        OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)


@pytest.mark.parametrize("hardness", ["hard", "prefer", "report"])
@pytest.mark.parametrize("selected_action", ["CA", None])
@pytest.mark.parametrize("best_alternative_winner", ["east", "north"])
@pytest.mark.parametrize("tactic", ["partner_enablement", "entry_unblocking"])
def test_native_paired_producer_compares_forced_root_continuations(rules, hardness, selected_action, best_alternative_winner, tactic):
    """Finite protocol graph; legal tail construction has a separate integration check."""
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Seat
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    roots = ("D7", "D8", "CA")
    nodes = []
    root_edges = []
    for index, (card, utility) in enumerate(zip(roots, (2, 1, -3), strict=True)):
        root_edges.append(ProofGraphEdgeV1.build(Card.parse(card).id, card, f"r{index}-1", play_trick_index=6))
        for depth, played in enumerate(("H7", "H8", "H9", "C7", "C8", "C9", "C10"), start=1):
            actor = ("south", "east", "north", "north", "west", "south", "east")[depth - 1]
            winner = "north" if depth == 3 or index == 0 else best_alternative_winner if index == 1 else "east"
            edge = ProofGraphEdgeV1.build(Card.parse(played).id, played, f"r{index}-{depth + 1}",
                play_trick_index=6 if depth < 4 else 7,
                **({"completed_trick_index": 6 if depth == 3 else 7,
                    "completed_trick_winner_seat": winner,
                    "completed_trick_winning_card_id": Card.parse("H9" if depth == 3 else "C10").id}
                   if depth in (3, 7) else {}))
            nodes.append(ProofGraphNodeV1.decision(f"r{index}-{depth}", Seat.parse(actor).team.name.lower(), (edge,), actor_seat=actor))
        nodes.append(ProofGraphNodeV1.leaf(f"r{index}-8", utility))
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "west_east", tuple(root_edges), actor_seat="west"), *nodes),
        tactic_context={"baseTricks": [3, 3], "trumpSuit": None, "ownerByCard": [1] * 32})
    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-PARTNER", "kind": "tactic.verified", "version": 1,
            "hardness": hardness, "arguments": {"kind": tactic, "minimumScore": 100,
                "minimumEvidence": 1, "causalRequired": True}},
        "decisionPoint": {"kind": "play", "selector": {}},
        "outcome": {"predicateId": "PRED-COUNT", "kind": "proof.winning_root_action_count", "version": 1,
            "hardness": "report", "arguments": {"minimum": 0, "maximum": 3}}})
    request = _request(selected_action=selected_action, claim="winning_root_action_count")
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={"contract": "sun", "trumpSuit": None}),
        graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
        tactical_detector_profile=TacticalDetectorProfileV1.defaults())
    proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
    assert proof.completion_status == "exact_complete"
    evidence = proof.native_qualification["PRED-PARTNER"]
    assert evidence["satisfied"] is (best_alternative_winner == "north")
    assert evidence["observed"]["bestAlternativeRootIndex"] == 1
    assert evidence["observed"]["root1Trick8WinnerSeat"] == int(Seat.parse(best_alternative_winner))
    assert evidence["observed"]["root2Trick8WinnerSeat"] == int(Seat.EAST)
    if tactic == "entry_unblocking":
        for index, card in enumerate(roots):
            assert evidence["observed"][f"root{index}Trick7LeaderSeat"] == int(Seat.WEST)
            assert evidence["observed"][f"root{index}Trick7LeadCard"] == Card.parse(card).id
        assert evidence["observed"]["root2Trick8LeaderSeat"] == int(Seat.NORTH)
        assert evidence["observed"]["root2Trick8LeadCard"] == Card.parse("C7").id
    assert len(proof.model_document()["pairedRawPayoffVariables"]) == 3


@pytest.mark.parametrize('hardness', ['hard', 'report', 'prefer'])
@pytest.mark.parametrize('tactic', ['small_card_big_gain', 'large_card_sacrifice'])
def test_native_configurer_does_not_upgrade_small_card_root_gain_to_causality(rules, hardness, tactic):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from dataclasses import replace
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    small = tactic == 'small_card_big_gain'
    original = _native_tactic_graph(winner='north' if small else 'west')
    graph = ProofGraphV1.build(scope=original.scope, root_node_id=original.root_node_id,
        nodes=tuple(replace(node, edges=tuple(replace(edge, action_id=15, action_code='HA')
            if node.node_id == 'root' and edge.action_code == ('H7' if small else 'S7') else edge for edge in node.edges))
            for node in original.nodes), tactic_context={**original.tactic_context, 'trumpSuit': None})
    scenario = ResearchScenarioV1.build({'schemaVersion': 1, 'origin': {'kind': 'search'},
        'constraints': {'predicateId': 'PRED-CAUSAL-GAIN', 'kind': 'tactic.verified', 'version': 1,
            'hardness': hardness, 'arguments': {'kind': tactic, 'minimumScore': 100,
                'minimumEvidence': 1, 'causalRequired': True}},
        'decisionPoint': {'kind': 'play', 'selector': {}},
        'outcome': {'predicateId': 'PRED-REPORT', 'kind': 'outcome.team_round', 'version': 1,
            'hardness': 'report', 'arguments': {'team': 'north_south', 'status': 'won'}}})
    request = _request(selected_action='S7' if small else 'HA')
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={'contract': 'sun', 'trumpSuit': None}),
        graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
        tactical_detector_profile=TacticalDetectorProfileV1.defaults())
    proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
    assert proof.completion_status == 'exact_complete' and proof.claim_satisfied
    evidence = proof.native_qualification['PRED-CAUSAL-GAIN']
    assert evidence['satisfied'] is False
    assert evidence['observed']['verified'] is True
    assert evidence['observed']['causalProven'] is False
    assert evidence['observed']['causalProofStatus'] == 'missing_native_delayed_payoff_witness'


@pytest.mark.parametrize('target_team', ['north_south', 'west_east'])
@pytest.mark.parametrize("selected_action", ["S7", None])
def test_native_tactic_target_seat_filters_known_absence_without_incomplete(rules, target_team, selected_action):
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    graph = _native_tactic_graph(winner='north')
    raw = _request(selected_action=selected_action, claim="winning_root_action_count").to_dict()
    raw.pop('sha256')
    raw['targetTeam'] = target_team
    request = ScenarioProofRequestV1.build(raw)
    for seat in ('north', 'west', 'south', 'east'):
        scenario = ResearchScenarioV1.build({'schemaVersion': 1, 'origin': {'kind': 'search'},
            'constraints': {'predicateId': 'PRED-TARGET', 'kind': 'tactic.applicable',
                'version': 1, 'hardness': 'hard',
                'arguments': {'kind': 'only_winning_move', 'targetSeat': seat, 'applicable': True}},
            'decisionPoint': {'kind': 'play', 'selector': {}},
            'outcome': {'predicateId': 'PRED-REPORT', 'kind': 'outcome.team_round',
                'version': 1, 'hardness': 'report', 'arguments': {'team': target_team, 'status': 'won'}}})
        configure = _native_proof_tactic_configurer_v3(scenario=scenario,
            candidate=SimpleNamespace(trace={}, state={'contract': 'sun'}), graph=graph,
            request=request, rules=rules, manifest=shared_scenario_predicate_catalog_lowering_manifest_v3())
        proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
        assert proof.completion_status == 'exact_complete', proof.solver_status
        assert proof.native_qualification['PRED-TARGET']['satisfied'] is (
            target_team == 'north_south' and seat == 'north')


@pytest.mark.parametrize("mode", ["exact", "bounded", "policy_conditioned"])
@pytest.mark.parametrize("actor,utilities,selected,expected", [
    ("north", (-1, 3), None, "SA"),
    ("west", (3, -1), None, "SA"),
    ("north", (3, 3), None, "S7"),
    ("west", (3, 3), None, "S7"),
    ("north", (-1, 3), "S7", "S7"),
    ("west", (3, -1), "S7", "S7"),
])
def test_native_principal_path_matches_canonical_root(mode, actor, utilities, selected, expected):
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Seat

    edges = tuple(ProofGraphEdgeV1.build(Card.parse(code).id, code, code,
        play_trick_index=6, completed_trick_index=6,
        completed_trick_winner_seat="north" if code == "SA" else "west",
        completed_trick_winning_card_id=Card.parse("SA" if code == "SA" else "SK").id)
        for code in ("S7", "SA"))
    graph = ProofGraphV1.build(scope="exact" if mode == "policy_conditioned" else mode, root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", Seat.parse(actor).team.name.lower(), edges, actor_seat=actor),
        *(ProofGraphNodeV1.leaf(code, value, kind="boundary" if mode == "bounded" else "terminal")
          for code, value in zip(("S7", "SA"), utilities, strict=True))),
        boundary_evaluator="current_raw_point_margin_v1" if mode == "bounded" else None,
        tactic_context={"baseTricks": [3, 3], "trumpSuit": None, "ownerByCard": [int(Seat.parse(actor))] * 32})

    def configure(bindings):
        card = bindings.principal_trick_winning_card_variables[0]
        seat = bindings.principal_trick_winner_seat_variables[0]
        literal = bindings.model.new_bool_var("principal_winner_is_ace")
        bindings.model.add(card == Card.parse("SA").id).only_enforce_if(literal)
        bindings.model.add(card != Card.parse("SA").id).only_enforce_if(literal.Not())
        return {"PATH": SimpleNamespace(literal=literal, extract=lambda solver: {
            "satisfied": solver.boolean_value(literal), "winningCardId": solver.value(card),
            "winnerSeat": solver.value(seat), "rootSelection": [solver.boolean_value(choice)
                for choice in bindings.root_selection_variables]})}

    proof = OrToolsScenarioGraphProofV1().prove(graph,
        _request(mode=mode, claim="winning_root_action_count", selected_action=selected,
            policies={seat: _policy(seat, []) for seat in ("north", "west", "south", "east")}
                if mode == "policy_conditioned" else None), tactic_configure=configure)
    assert proof.completion_status == {"exact": "exact_complete", "bounded": "bounded_complete",
                                      "policy_conditioned": "policy_complete"}[mode]
    assert proof.selected_action == expected
    assert proof.principal_variation == (expected,)
    evidence = proof.native_qualification["PATH"]
    assert evidence["winningCardId"] == Card.parse("SA" if expected == "SA" else "SK").id
    assert evidence["winnerSeat"] == int(Seat.NORTH if expected == "SA" else Seat.WEST)
    assert evidence["satisfied"] is (expected == "SA")
    assert evidence["rootSelection"] == [expected == "S7", expected == "SA"]


@pytest.mark.parametrize("delayed_control", [False, True])
def test_native_paired_raw_paths_force_each_root_across_a_transposition(rules, delayed_control):
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Contract

    # Abstract shared-suffix graph; legal native tails have a separate regression.
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", tuple(
            ProofGraphEdgeV1.build(Card.parse(card).id, card, "shared", play_trick_index=7)
            for card in ("DA", "C10")), actor_seat="north"),
        ProofGraphNodeV1.decision("shared", "west_east", (
            ProofGraphEdgeV1.build(Card.parse("SA").id, "SA", "end", play_trick_index=7,
                completed_trick_index=7, completed_trick_winner_seat="west",
                completed_trick_winning_card_id=Card.parse("SA").id),), actor_seat="west"),
        ProofGraphNodeV1.leaf("end", 0)),
        tactic_context={"baseTricks": [3, 4], "trumpSuit": None, "ownerByCard": [0] * 32})
    points = tuple(rules.play_domain.points(Card(card), Contract.SUN, None) for card in range(32))
    prefix = (Card.parse("S7").id, Card.parse("S8").id)

    def configure(bindings):
        with pytest.raises(ValueError, match="native point tables"):
            bindings.bind_root_raw_payoffs((True, *points[1:]), 10, prefix)
        with pytest.raises(ValueError, match="native point tables"):
            bindings.bind_root_raw_payoffs(points, 10, (prefix[0], prefix[0]))
        for leader in (True, -1, 4):
            with pytest.raises(ValueError, match="native point tables"):
                bindings.bind_root_raw_payoffs(points, 10, prefix, first_leader=leader)
        with pytest.raises(ValueError, match="native point tables"):
            bindings.bind_root_raw_payoffs(points, 10, prefix, delayed_control=1)
        with pytest.raises(ValueError, match="native point tables"):
            bindings.bind_root_raw_payoffs(points, 10, prefix, delayed_control=True)
        rows = bindings.bind_root_raw_payoffs(points, 10, prefix,
            **({"first_leader": 2, "delayed_control": True} if delayed_control else {}))
        with pytest.raises(ValueError, match="already bound"):
            bindings.bind_root_raw_payoffs(points, 10, prefix)
        return {"PAIRS": SimpleNamespace(literal=bindings.proof_complete_variable,
            extract=lambda solver: [{key: solver.value(value) for key, value in row.items()} for row in rows])}

    request = _request(selected_action="C10", claim="winning_root_action_count")
    service = OrToolsScenarioGraphProofV1()
    plain = service.prove(graph, request)
    proof = service.prove(graph, request, tactic_configure=configure)
    assert proof.completion_status == "exact_complete" and proof.selected_action == "C10"
    assert proof.root_actions == plain.root_actions and proof.principal_variation == plain.principal_variation
    assert "pairedRawPayoffVariables" not in plain.model_document()
    rows = proof.native_qualification["PAIRS"]
    if delayed_control:
        assert all(row["trick8WinningCard"] == Card.parse("SA").id for row in rows)
        assert all(row["trick8OpponentTrumpsRemaining"] == 0 for row in rows)
        assert [row["trick8CardRawMargin"] for row in rows] == [-22, -21]
        assert all(pair["delayedControl"] is True for pair in proof.model_document()["pairedRawPayoffVariables"])
    else:
        assert all(not any(key.endswith(("LeaderSeat", "LeadCard", "CardRawMargin", "WinningCard",
            "OpponentTrumpsRemaining")) for key in row) for row in rows)
    assert [row["continuationRawMargin"] for row in rows] == [-32, -31]
    assert all(row["immediateRawMargin"] == 0 for row in rows)
    stopped = [False]

    def cancel_during_binding(bindings):
        stopped[0] = True
        bindings.bind_root_raw_payoffs(points, 10, prefix)

    with pytest.raises(InterruptedError, match="cancelled"):
        service.prove(graph, request, tactic_configure=cancel_during_binding,
                      cancel_check=lambda: stopped[0])


def test_native_paired_delayed_control_counts_only_unplayed_opponent_trumps(rules):
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Contract, Seat, Suit

    # Abstract branches exercise the path bindings; legal reachability is tested separately.
    roots, nodes = [], []
    for index, cards in enumerate((("H7", "SA", "H9", "H10", "C7", "C8", "C9", "C10"),
                                    ("H8", "SK", "H9", "H10", "SA", "C8", "C9", "C10"))):
        for depth, (card, actor) in enumerate(zip(cards, ("north", "west", "south", "east",
                                                         "west", "south", "east", "north"), strict=True)):
            edge = ProofGraphEdgeV1.build(Card.parse(card).id, card, f"p{index}-{depth + 1}",
                play_trick_index=6 + depth // 4,
                **({"completed_trick_index": 6 + depth // 4,
                    "completed_trick_winner_seat": "west" if depth == 3 or index == 1 else "north",
                    "completed_trick_winning_card_id": Card.parse(cards[1] if depth == 3 else
                        cards[4] if index == 1 else "C10").id} if depth in (3, 7) else {}))
            if depth == 0:
                roots.append(edge)
            else:
                nodes.append(ProofGraphNodeV1.decision(f"p{index}-{depth}", Seat.parse(actor).team.name.lower(),
                    (edge,), actor_seat=actor))
        nodes.append(ProofGraphNodeV1.leaf(f"p{index}-8", 3 - index))
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", tuple(roots), actor_seat="north"), *nodes),
        tactic_context={"baseTricks": [3, 3], "trumpSuit": int(Suit.SPADES), "ownerByCard": [0] * 32})
    points = tuple(rules.play_domain.points(Card(card), Contract.HOKUM, Suit.SPADES) for card in range(32))

    def configure(bindings):
        rows = bindings.bind_root_raw_payoffs(points, 10, (), first_leader=0, delayed_control=True)
        return {"PAIRS": SimpleNamespace(literal=bindings.proof_complete_variable,
            extract=lambda solver: [{key: solver.value(value) for key, value in row.items()} for row in rows])}

    proof = OrToolsScenarioGraphProofV1().prove(graph,
        _request(selected_action="H7", claim="winning_root_action_count"), tactic_configure=configure)
    assert proof.completion_status == "exact_complete"
    rows = proof.native_qualification["PAIRS"]
    assert [row["trick7WinningCard"] for row in rows] == [Card.parse("SA").id, Card.parse("SK").id]
    assert [row["trick7OpponentTrumpsRemaining"] for row in rows] == [0, 1]
    assert [row["trick8OpponentTrumpsRemaining"] for row in rows] == [0, 0]


def test_native_tactic_literal_uses_same_solve_root_and_principal_path_vars() -> None:
    request = _request(selected_action="S7")

    def configure(bindings):
        winner_is_target = bindings.model.new_bool_var("winner_is_target")
        bindings.model.add(
            bindings.first_principal_trick_winner_team_variable == 0
        ).only_enforce_if(winner_is_target)
        bindings.model.add(
            bindings.first_principal_trick_winner_team_variable != 0
        ).only_enforce_if(winner_is_target.Not())
        literal = bindings.model.new_bool_var("native_tactic_literal")
        bindings.model.add_bool_and(
            (
                bindings.root_winning_variables[
                    bindings.selected_root_action_index
                ],
                bindings.principal_variation_has_continuation_variable,
                winner_is_target,
            )
        ).only_enforce_if(literal)
        bindings.model.add_bool_or(
            (
                bindings.root_winning_variables[
                    bindings.selected_root_action_index
                ].Not(),
                bindings.principal_variation_has_continuation_variable.Not(),
                winner_is_target.Not(),
            )
        ).only_enforce_if(literal.Not())
        return {
            "PRED-NATIVE-TACTIC": SimpleNamespace(
                literal=literal,
                extract=lambda solver: {
                    "satisfied": solver.boolean_value(literal)
                },
            )
        }

    positive = OrToolsScenarioGraphProofV1().prove(
        _native_tactic_graph(winner="north"),
        request,
        tactic_configure=configure,
    )
    counterexample = OrToolsScenarioGraphProofV1().prove(
        _native_tactic_graph(winner="west"),
        request,
        tactic_configure=configure,
    )

    assert positive.native_qualification is not None
    assert positive.native_qualification["PRED-NATIVE-TACTIC"]["satisfied"] is True
    assert counterexample.native_qualification is not None
    assert (
        counterexample.native_qualification["PRED-NATIVE-TACTIC"]["satisfied"]
        is False
    )


@pytest.mark.parametrize('prefix_length', range(4))
@pytest.mark.parametrize('requested_count', [0, 1])
def test_native_principal_trick_count_includes_already_played_cards(rules, prefix_length, requested_count):
    from baloot_platform.core.cards import Card
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    plays = [('north', 'C8'), ('west', 'C10'), ('south', 'D7'), ('east', 'DQ')]
    nodes = []
    for index in range(prefix_length, 4):
        seat, code = plays[index]
        nodes.append(ProofGraphNodeV1.decision(str(index),
            'north_south' if seat in {'north', 'south'} else 'west_east',
            (ProofGraphEdgeV1.build(Card.parse(code).id, code, str(index + 1), play_trick_index=7,
                **(dict(completed_trick_index=7, completed_trick_winner_seat='west',
                        completed_trick_winning_card_id=Card.parse('C10').id) if index == 3 else {})),),
            actor_seat=seat))
    graph = ProofGraphV1.build(scope='exact', root_node_id=str(prefix_length),
        nodes=(*nodes, ProofGraphNodeV1.leaf('4', 3)),
        tactic_context={'baseTricks': [3, 4], 'trumpSuit': None, 'ownerByCard': [0] * 32})
    request = _request(selected_action=plays[prefix_length][1])
    scenario = ResearchScenarioV1.build(dict(schemaVersion=1, origin={'kind': 'search'},
        constraints=dict(predicateId='PRED-COUNT', kind='proof.principal_variation_trick_team_rank_count',
            version=1, hardness='hard', arguments={'trickIndex': 7, 'team': 'north_south', 'rank': ['seven'],
                                                   'minimum': requested_count, 'maximum': requested_count}),
        decisionPoint={'kind': 'play', 'selector': {}},
        outcome=dict(predicateId='PRED-REPORT', kind='outcome.team_round', version=1,
                     hardness='report', arguments={'team': 'north_south', 'status': 'won'})))
    candidate = SimpleNamespace(trace={}, state={'contract': 'sun', 'currentTrick': {'index': 7,
        'plays': [dict(actor=seat, card=code) for seat, code in plays[:prefix_length]]}})
    configure = _native_proof_tactic_configurer_v3(scenario=scenario, candidate=candidate,
        graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3())
    proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure)
    evidence = proof.native_qualification['PRED-COUNT']
    assert evidence['observed']['matchingCount'] == 1
    assert evidence['satisfied'] is (requested_count == 1)


@pytest.mark.parametrize("neutral", [False, True])
def test_project_counterfactual_recomputes_both_teams_strategy(neutral):
    from dataclasses import replace

    source = _graph()
    graph = ProofGraphV1.build(
        scope="exact", root_node_id=source.root_node_id,
        nodes=[replace(node, actor_seat="north") if node.node_id == "root" else node
               for node in source.nodes],
        tactic_context=_native_tactic_graph(winner="north").tactic_context,
    )
    utilities = {"five": -2, "one": 0, "three-a": -8, "three-b": -4}
    alternate = ProofGraphV1.build(
        scope="exact", root_node_id=graph.root_node_id,
        nodes=[replace(node, utility=utilities[node.node_id])
               if node.is_leaf and not neutral else node for node in graph.nodes],
        tactic_context=graph.tactic_context,
    )

    def configure(bindings):
        literal = bindings.model.new_bool_var("project_gain_positive")
        delta = bindings.root_project_consequence_variables[1]
        bindings.model.add(delta > 0).only_enforce_if(literal)
        bindings.model.add(delta <= 0).only_enforce_if(literal.Not())
        return {"PROJECT": SimpleNamespace(literal=literal, extract=lambda solver: {
            "satisfied": solver.boolean_value(literal),
            "deltas": [solver.value(item) for item in bindings.root_project_consequence_variables],
        })}

    service = OrToolsScenarioGraphProofV1()
    result = service.prove(graph, _request(), tactic_configure=configure,
                           native_root_only=True, project_counterfactual_graph=alternate)
    assert result.native_qualification["PROJECT"] == {
        "satisfied": not neutral, "deltas": [0, 0] if neutral else [3, 7],
    }
    assert result.root_actions == service.prove(graph, _request()).root_actions
    assert result.model_document()["projectCounterfactualGraph"] == alternate.to_dict()
    assert canonical_sha256(result.model_document()) == result.model_hash
    changed_choices = ProofGraphV1.build(
        scope="exact", root_node_id=alternate.root_node_id,
        nodes=[replace(node, actor_team="north_south") if node.node_id == "west" else node
               for node in alternate.nodes], tactic_context=alternate.tactic_context,
    )
    with pytest.raises(ValueError, match="preserve the complete legal graph"):
        service.prove(graph, _request(), tactic_configure=configure,
                      project_counterfactual_graph=changed_choices)


@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("selected_action", ["declare", None])
@pytest.mark.parametrize("hardness", ["hard", "report", "prefer"])
@pytest.mark.parametrize("minimum_gain,scale", [(4.0, 1), (4.01, 1), (4.0, 1000), (4.0001, 1000)])
def test_executor_project_qualifier_consumes_paired_solver_values(rules, neutral, selected_action, hardness, minimum_gain, scale):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-PROJECT", "kind": "tactic.verified", "version": 1,
            "hardness": hardness, "arguments": {"kind": "project_bidding", "minimumScore": 100,
                "minimumEvidence": 1, "causalRequired": True}},
        "decisionPoint": {"kind": "project", "selector": {"owner": "north"}},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round", "version": 1,
            "hardness": "report", "arguments": {"team": "north_south", "status": "won"}}})
    root = ProofGraphNodeV1.decision("root", "north_south", (
        _edge(0, "pass", "pass-end"), _edge(1, "declare", "declare-end")), actor_seat="north")
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", utility_scale=scale, nodes=(root,
        ProofGraphNodeV1.leaf("pass-end", -scale), ProofGraphNodeV1.leaf("declare-end", 3 * scale)))
    counter = ProofGraphV1.build(scope="exact", root_node_id="root", utility_scale=scale, nodes=(root,
        ProofGraphNodeV1.leaf("pass-end", -scale), ProofGraphNodeV1.leaf("declare-end", (3 if neutral else -1) * scale)))
    request = _request(selected_action=selected_action, claim="winning_root_action_count")
    profile_raw = TacticalDetectorProfileV1.defaults().to_dict()
    profile_raw.pop("sha256")
    profile = TacticalDetectorProfileV1.build({**profile_raw, "projectConsequenceDelta": minimum_gain})
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={}), graph=graph,
        request=request, rules=rules, manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
        tactical_detector_profile=profile)
    outcome = OrToolsScenarioGraphProofV1().prove(graph, request, native_root_only=True,
        tactic_configure=configure, project_counterfactual_graph=counter)
    assert outcome.native_qualification["PRED-PROJECT"]["satisfied"] is (not neutral and minimum_gain <= 4)
    evidence = outcome.native_qualification["PRED-PROJECT"]["observed"]
    assert evidence["qualificationSource"] == "native_cp_sat_paired_project_consequence_variables"
    assert evidence["detectorVersion"] == "3.0.1"
    assert evidence["detectorProfileHash"] == profile.sha256
    assert evidence["projectConsequenceDelta"] == minimum_gain
    assert evidence["projectConsequenceUnit"] == "awarded_point_margin"
    assert evidence["projectConsequenceScale"] == scale
    assert evidence["rootProjectConsequence0"] == 0
    assert evidence["rootProjectConsequence1"] == (0 if neutral else 4 * scale)


@pytest.mark.parametrize("line_count", [False, True])
def test_auction_root_count_qualifier_needs_no_solved_contract(rules, line_count):
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-COUNT", "kind": "proof.winning_continuation_line_count" if line_count else "proof.winning_root_action_count",
            "version": 1, "hardness": "hard", "arguments": {"minimum": "1" if line_count else 1, "maximum": "1" if line_count else 1,
                **({"lineUniverse": "all_legal"} if line_count else {})}},
        "decisionPoint": {"kind": "bid", "selector": {}},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round", "version": 1,
            "hardness": "report", "arguments": {"team": "north_south", "status": "won"}}})
    graph = ProofGraphV1.build(scope="exact", root_node_id="bid", nodes=(
        ProofGraphNodeV1.decision("bid", "north_south", (
            _edge(0, "pass", "lose"), _edge(1, "sun", "win")), actor_seat="north"),
        ProofGraphNodeV1.leaf("lose", -1), ProofGraphNodeV1.leaf("win", 3)))
    request = _request(selected_action="sun", line_universe="all_legal" if line_count else None)
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={}), graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3())
    outcome = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure,
        native_root_only=True, native_line_count=line_count)
    assert outcome.native_qualification["PRED-COUNT"]["satisfied"]
    observed = outcome.native_qualification["PRED-COUNT"]["observed"]
    assert observed["exactCount"] == "1" if line_count else observed["winningCount"] == 1


@pytest.mark.parametrize("decision_kind", ["bid", "escalation"])
@pytest.mark.parametrize("predicate_kind", ["tactic.applicable", "tactic.verified"])
@pytest.mark.parametrize("hardness", ["hard", "report"])
@pytest.mark.parametrize("selected_action", ["sun", None])
@pytest.mark.parametrize("tactic,values,expected,automatic_expected", [
    ("only_winning_move", (-1, 3), True, True),
    ("only_winning_move", (3, -1), False, True),
    ("contract_rescue", (-1, 1), True, True),
    ("contract_rescue", (1, -1), False, True),
    ("debate", (3, 3), True, True),
    ("debate", (-1, 3), False, False),
])
def test_nonplay_root_tactics_need_no_solved_contract(rules, decision_kind, predicate_kind, hardness, selected_action, tactic, values, expected, automatic_expected):
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-TACTIC", "kind": predicate_kind,
            "version": 1, "hardness": hardness, "arguments": {"kind": tactic,
                "actingSeat": "north", "observer": "north", "targetTeam": "north_south",
                **({"applicable": True} if predicate_kind == "tactic.applicable" else
                   {"minimumScore": 100, "minimumEvidence": 1, "causalRequired": tactic != "debate"})}},
        "decisionPoint": {"kind": decision_kind, "selector": {}},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "outcome.team_round", "version": 1,
            "hardness": "report", "arguments": {"team": "north_south", "status": "won"}}})
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            _edge(0, "pass", "first"), _edge(1, "sun", "second")), actor_seat="north"),
        ProofGraphNodeV1.leaf("first", values[0]), ProofGraphNodeV1.leaf("second", values[1])))
    raw = _request(selected_action=selected_action, claim="outcome_reachable").to_dict()
    raw.pop("sha256")
    if tactic == "contract_rescue":
        raw.update(utility="contract_success", successThreshold=1)
    request = ScenarioProofRequestV1.build(raw)
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={}), graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3())
    outcome = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure, native_root_only=True)
    assert outcome.completion_status == "exact_complete"
    evidence = outcome.native_qualification["PRED-TACTIC"]
    assert evidence["satisfied"] is (expected if selected_action is not None else automatic_expected)
    assert evidence["observed"]["selectedRootAction"] == outcome.selected_action
    assert evidence["observed"]["qualificationSource"].startswith("native_cp_sat_root_")


def test_solver_owned_tactic_survives_same_solve_and_result_projection(rules):
    """Exercise the native evidence protocol, not full-round reachability."""
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import (
        OrToolsResearchStageReceiptV2,
        _native_proof_tactic_configurer_v3,
        _native_same_solve_evaluation_v3,
        _proof_receipt_evidence_bindings_v3,
    )
    from baloot_platform.research.result_authority import _native_tactic_projection
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-TACTIC", "kind": "tactic.verified",
            "version": 1, "hardness": "hard", "arguments": {"kind": "only_winning_move",
                "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True}},
        "decisionPoint": {"kind": "bid", "selector": {}},
        "outcome": {"predicateId": "PRED-WIN", "kind": "proof.winning_root_action_count",
            "version": 1, "hardness": "hard", "arguments": {"minimum": 1, "maximum": 1}}})
    graph = ProofGraphV1.build(scope="exact", root_node_id="bid", nodes=(
        ProofGraphNodeV1.decision("bid", "north_south", (
            _edge(0, "pass", "lose"), _edge(1, "sun", "win")), actor_seat="north"),
        ProofGraphNodeV1.leaf("lose", -1), ProofGraphNodeV1.leaf("win", 3)))
    request = _request(selected_action="sun", claim="outcome_reachable")
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={}), graph=graph, request=request, rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3())
    proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure, native_root_only=True)
    assert proof.completion_status == "exact_complete" and proof.claim_satisfied
    receipt = OrToolsResearchStageReceiptV2(stage="proof", model_hash=proof.model_hash,
        evidence_hash=proof.sha256, semantic_hash=proof.sha256, solver_status=proof.solver_status,
        evidence_bindings=_proof_receipt_evidence_bindings_v3(proof, request))
    candidate = SimpleNamespace(trace={}, stage_receipts={"proof": receipt})
    evaluation = _native_same_solve_evaluation_v3(scenario, candidate, proof, {})
    assert evaluation is not None and evaluation.satisfied
    evidence = next(row for row in evaluation.evidence if row["predicateId"] == "PRED-TACTIC")
    assert evidence["evidenceExtractor"] == "cp_sat.native_proof_same_solve_v1"
    assert evidence["observed"] == proof.native_qualification["PRED-TACTIC"]
    assert _native_tactic_projection(scenario.to_dict(), list(evaluation.evidence),
        proof_outcome=proof.to_dict()) == (["only_winning_move"], ["only_winning_move"])


@pytest.mark.parametrize("decision_kind", ["bid", "escalation"])
@pytest.mark.parametrize("predicate_kind", ["tactic.applicable", "tactic.verified"])
@pytest.mark.parametrize("mode", ["exact", "bounded"])
@pytest.mark.parametrize("tactic,target,expected,selected", [
    ("kabout_creation", "north_south", True, "sun"),
    ("kabout_preservation", "north_south", False, "sun"),
    ("kabout_prevention", "west_east", True, "pass"),
])
def test_preplay_kabout_tactics_bind_zero_completed_tricks(
    rules, decision_kind, predicate_kind, mode, tactic, target, expected, selected,
):
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-KABOUT", "kind": predicate_kind,
            "version": 1, "hardness": "hard", "arguments": {"kind": tactic,
                "actingSeat": "north", "targetTeam": target,
                **({"applicable": expected} if predicate_kind == "tactic.applicable" else
                   {"minimumScore": 100 if expected else 0, "minimumEvidence": 1 if expected else 0,
                    "causalRequired": expected})}},
        "decisionPoint": {"kind": decision_kind, "selector": {}},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "state.phase", "version": 1,
            "hardness": "report", "arguments": {"state": "playing"}}})
    graph = ProofGraphV1.build(scope=mode, root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            _edge(0, "pass", "lose"), _edge(1, "sun", "win")), actor_seat="north"),
        ProofGraphNodeV1.leaf("lose", -1), ProofGraphNodeV1.leaf("win", 1)),
        boundary_evaluator="current_raw_point_margin_v1" if mode == "bounded" else None)
    raw = _request(mode=mode, selected_action=None, claim="winning_root_action_count").to_dict()
    raw.pop("sha256")
    raw.update(utility="kabout", successThreshold=1, targetTeam=target)
    if mode == "bounded":
        raw["horizon"] = {"unit": "round_end", "value": 1}
    request = ScenarioProofRequestV1.build(raw)
    candidate = SimpleNamespace(trace={}, state={"phase": "bidding" if decision_kind == "bid"
        else "final_dealt", "completedTricks": [], "currentTrick": None})
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
    configure = _native_proof_tactic_configurer_v3(scenario=scenario, candidate=candidate,
        graph=graph, request=request, rules=rules,
        manifest=manifest)
    outcome = OrToolsScenarioGraphProofV1().prove(
        graph, request, tactic_configure=configure, native_root_only=True)
    assert outcome.completion_status == ("bounded_complete" if mode == "bounded" else "exact_complete")
    observed = outcome.native_qualification["PRED-KABOUT"]["observed"]
    assert observed["selectedRootAction"] == selected == outcome.selected_action
    assert observed["applicable"] is expected
    assert observed["verified"] is expected
    assert outcome.native_qualification["PRED-KABOUT"]["satisfied"] is (
        True if predicate_kind == "tactic.applicable" else expected)
    if tactic == "kabout_creation" and predicate_kind == "tactic.applicable":
        for state in (
            {**candidate.state, "phase": "playing"},
            {**candidate.state, "completedTricks": [{}]},
            {**candidate.state, "completedTricks": ()},
            {**candidate.state, "currentTrick": {}},
            {key: value for key, value in candidate.state.items() if key != "currentTrick"},
            {key: value for key, value in candidate.state.items() if key != "completedTricks"},
        ):
            with pytest.raises(ValueError, match="unplayed normalized state"):
                _native_proof_tactic_configurer_v3(scenario=scenario,
                    candidate=SimpleNamespace(trace={}, state=state), graph=graph,
                    request=request, rules=rules, manifest=manifest)


@pytest.mark.parametrize("hardness", ["prefer", "report"])
@pytest.mark.parametrize("tactic", ["kabout_creation", "kabout_preservation", "kabout_prevention"])
def test_kabout_reports_do_not_require_kabout_bindings_for_other_utilities(rules, hardness, tactic):
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    from baloot_platform.research.ortools_research_executor import _native_proof_tactic_configurer_v3
    from baloot_platform.research.scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3

    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-KABOUT-REPORT", "kind": "tactic.applicable",
            "version": 1, "hardness": hardness, "arguments": {"kind": tactic, "applicable": False}},
        "decisionPoint": {"kind": "bid", "selector": {}},
        "outcome": {"predicateId": "PRED-REPORT", "kind": "state.phase", "version": 1,
            "hardness": "report", "arguments": {"state": "playing"}}})
    assert _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={}), graph=_native_tactic_graph(winner="north"),
        request=_request(selected_action=None, claim="winning_root_action_count"), rules=rules,
        manifest=shared_scenario_predicate_catalog_lowering_manifest_v3()) is None


def test_root_only_bindings_preserve_proof_without_principal_model() -> None:
    sizes = []

    def configure(bindings):
        sizes.append(len(bindings.model.proto.constraints))
        literal = bindings.root_winning_variables[0]
        return {"ROOT": SimpleNamespace(
            literal=literal,
            extract=lambda solver: {"satisfied": solver.boolean_value(literal)},
        )}

    prover = OrToolsScenarioGraphProofV1()
    graph = _native_tactic_graph(winner="north")
    request = _request(selected_action="S7")
    full = prover.prove(graph, request, tactic_configure=configure)
    root = prover.prove(graph, request, tactic_configure=configure, native_root_only=True)
    assert root.root_actions == full.root_actions
    assert root.principal_variation == full.principal_variation
    assert root.claim_satisfied == full.claim_satisfied
    assert root.native_qualification == full.native_qualification
    assert sizes[1] < sizes[0]
    noncard = ProofGraphV1.build(scope=graph.scope, root_node_id=graph.root_node_id,
        nodes=graph.nodes)
    def noncard_bindings(bindings):
        assert bindings.root_card_variables == bindings.root_owner_variables == ()
        assert bindings.base_trick_variables == ()
        assert bindings.trump_suit_variable is None
        return configure(bindings)
    noncard_result = prover.prove(noncard, request, native_root_only=True,
        tactic_configure=noncard_bindings)
    assert noncard_result.root_actions == root.root_actions
    assert noncard_result.native_qualification == root.native_qualification
    with pytest.raises(ValueError, match="tactic bindings"):
        prover.prove(noncard, request, tactic_configure=configure)
    for invalid in (None, 1, "true"):
        with pytest.raises(TypeError, match="native_root_only"):
            prover.prove(graph, request, native_root_only=invalid)
    with pytest.raises(ValueError, match="root-only"):
        prover.prove(graph, request, native_root_only=True)


def _policy(seat: str, choices: list[dict], *, seed: int = 0) -> dict:
    ordered = sorted(choices, key=lambda item: item["nodeId"])
    core = {
        "schemaVersion": 1,
        "seat": seat,
        "policyId": "fixed_action_map_v1",
        "version": "1.0.0",
        "seed": seed,
        "choices": ordered,
    }
    return {
        "schemaVersion": 1,
        "policyId": "fixed_action_map_v1",
        "version": "1.0.0",
        "seed": seed,
        "choices": ordered,
        "profileHash": canonical_sha256(core),
    }


def _policies(*, west_action: int) -> dict:
    return {
        "north": _policy("north", []),
        "west": _policy(
            "west",
            [{"nodeId": "reply", "actionId": west_action}],
            seed=7,
        ),
        "south": _policy("south", []),
        "east": _policy("east", []),
    }


def _policy_templates() -> dict:
    profiles = {}
    for index, seat in enumerate(("north", "west", "south", "east"), start=1):
        core = {
            "schemaVersion": 1,
            "seat": seat,
            "policyId": "fixed_action_map_v1",
            "version": "1.0.0",
            "seed": index,
            "choices": [],
            "materialization": "seeded_legal_action_order_v1",
        }
        profiles[seat] = {
            key: value for key, value in core.items() if key != "seat"
        }
        profiles[seat]["profileHash"] = canonical_sha256(core)
    return profiles


def test_cp_sat_values_root_count_and_stable_principal_variation() -> None:
    proof = OrToolsScenarioGraphProofV1().prove(_graph(), _request())

    assert proof.completion_status == "exact_complete"
    assert proof.solver_status == "optimal"
    assert proof.claim_satisfied is True
    assert proof.winning_root_action_count == 1
    assert [(item.action_code, item.value, item.satisfies) for item in proof.root_actions] == [
        ("A", 1, False),
        ("B", 3, True),
    ]
    assert proof.principal_variation == ("B", "N2")


def test_graph_and_proof_hashes_ignore_input_order() -> None:
    request = _request()
    first_graph = _graph()
    second_graph = _graph(reverse=True)
    first = OrToolsScenarioGraphProofV1().prove(first_graph, request)
    second = OrToolsScenarioGraphProofV1().prove(second_graph, request)

    assert first_graph.sha256 == second_graph.sha256
    assert first.model_hash == second.model_hash
    assert first.sha256 == second.sha256


def test_false_claim_is_not_a_model_error() -> None:
    proof = OrToolsScenarioGraphProofV1().prove(
        _graph(),
        _request(selected_action="A"),
    )

    assert proof.completion_status == "proof_failed"
    assert proof.solver_status == "optimal"
    assert proof.claim_satisfied is False
    assert proof.winning_root_action_count == 1
    assert proof.root_actions


def _with_native_count_context(graph: ProofGraphV1) -> ProofGraphV1:
    root = next(node for node in graph.nodes if node.node_id == graph.root_node_id)
    return ProofGraphV1.build(
        scope=graph.scope,
        root_node_id=graph.root_node_id,
        nodes=(
            ProofGraphNodeV1.decision(
                root.node_id, root.actor_team, root.edges, actor_seat="north"
            ),
            *(node for node in graph.nodes if node.node_id != root.node_id),
        ),
        tactic_context={"baseTricks": [0, 0], "trumpSuit": None, "ownerByCard": [0] * 32},
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (cp_model.INFEASIBLE, "model_error"),
        (cp_model.FEASIBLE, "proof_incomplete"),
    ],
)
def test_infeasible_is_model_error_and_feasible_is_not_exact(
    monkeypatch: pytest.MonkeyPatch,
    status,
    expected: str,
) -> None:
    class FakeSolver:
        @staticmethod
        def status_name(_status) -> str:
            return {
                cp_model.INFEASIBLE: "INFEASIBLE",
                cp_model.FEASIBLE: "FEASIBLE",
            }[_status]

    monkeypatch.setattr(
        OrToolsScenarioGraphProofV1,
        "_solve",
        staticmethod(lambda _model, maximum_seconds: (FakeSolver(), status)),
    )

    proof = OrToolsScenarioGraphProofV1().prove(_graph(), _request())

    assert proof.completion_status == expected
    assert proof.claim_satisfied is False
    assert proof.winning_root_action_count is None
    assert proof.root_actions == ()


def test_bounded_graph_reports_bounded_completion() -> None:
    proof = OrToolsScenarioGraphProofV1().prove(
        _graph(scope="bounded"),
        _request(mode="bounded"),
    )

    assert proof.completion_status == "bounded_complete"
    assert proof.winning_root_action_count == 1


@pytest.mark.parametrize("mode", ("exact", "bounded"))
def test_root_action_count_is_an_explicit_complete_state_claim(mode: str) -> None:
    proof = OrToolsScenarioGraphProofV1().prove(
        _graph(scope=mode),
        _request(
            mode=mode,
            claim="winning_root_action_count",
            selected_action=None,
        ),
    )

    assert proof.completion_status == f"{mode}_complete"
    assert proof.claim_satisfied is True
    assert proof.winning_root_action_count == 1


def test_graph_rejects_cycles_and_unreachable_nodes() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        ProofGraphV1.build(
            scope="exact",
            root_node_id="a",
            nodes=(
                ProofGraphNodeV1.decision(
                    "a", "north_south", (_edge(1, "A", "b"),)
                ),
                ProofGraphNodeV1.decision(
                    "b", "west_east", (_edge(2, "B", "a"),)
                ),
            ),
        )

    with pytest.raises(ValueError, match="unreachable"):
        ProofGraphV1.build(
            scope="exact",
            root_node_id="root",
            nodes=(
                ProofGraphNodeV1.decision(
                    "root", "north_south", (_edge(1, "A", "end"),)
                ),
                ProofGraphNodeV1.leaf("end", 1),
                ProofGraphNodeV1.leaf("orphan", 0),
            ),
        )


def test_auto_selection_uses_smallest_action_id_on_equal_values() -> None:
    graph = ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                "north_south",
                (_edge(9, "HIGH-ID", "a"), _edge(2, "LOW-ID", "b")),
            ),
            ProofGraphNodeV1.leaf("a", 4),
            ProofGraphNodeV1.leaf("b", 4),
        ),
    )
    request = _request(claim="outcome_forced", selected_action=None)

    proof = OrToolsScenarioGraphProofV1().prove(graph, request)

    assert proof.selected_action == "LOW-ID"
    assert proof.principal_variation == ("LOW-ID",)


def test_all_legal_and_optimal_strategy_count_different_winning_paths() -> None:
    all_legal = OrToolsScenarioGraphProofV1().prove(
        _graph(),
        _request(
            claim="winning_continuation_line_count",
            selected_action="A",
            line_universe="all_legal",
        ),
    )
    optimal = OrToolsScenarioGraphProofV1().prove(
        _graph(),
        _request(
            claim="winning_continuation_line_count",
            selected_action="A",
            line_universe="optimal_strategy",
        ),
    )

    assert all_legal.continuation_lines is not None
    assert all_legal.continuation_lines.complete is True
    assert all_legal.continuation_lines.exact_count == "1"
    assert optimal.continuation_lines is not None
    assert optimal.continuation_lines.complete is True
    assert optimal.continuation_lines.exact_count == "0"


def test_transposition_edges_retain_path_multiplicity() -> None:
    graph = ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root", "north_south", (_edge(1, "ROOT", "fork"),)
            ),
            ProofGraphNodeV1.decision(
                "fork",
                "west_east",
                (_edge(2, "LEFT", "win"), _edge(3, "RIGHT", "win")),
            ),
            ProofGraphNodeV1.leaf("win", 3),
        ),
    )
    request = _request(
        claim="winning_continuation_line_count",
        selected_action="ROOT",
        line_universe="optimal_strategy",
    )

    proof = OrToolsScenarioGraphProofV1().prove(graph, request)

    assert proof.continuation_lines is not None
    assert proof.continuation_lines.exact_count == "2"


@pytest.mark.parametrize(
    ("graph", "selected", "universe", "expected"),
    [
        (_graph(), "A", "all_legal", 1),
        (_graph(), "A", "optimal_strategy", 0),
        (
            ProofGraphV1.build(
                scope="exact",
                root_node_id="root",
                nodes=(
                    ProofGraphNodeV1.decision("root", "north_south", (_edge(1, "ROOT", "fork"),)),
                    ProofGraphNodeV1.decision("fork", "west_east", (
                        _edge(2, "LEFT", "win"), _edge(3, "RIGHT", "win"),
                    )),
                    ProofGraphNodeV1.leaf("win", 3),
                ),
            ),
            "ROOT",
            "optimal_strategy",
            2,
        ),
    ],
)
def test_native_same_solve_line_count_matches_independent_count(
    graph, selected, universe, expected, monkeypatch
) -> None:
    native_graph = _with_native_count_context(graph)
    request = _request(
        claim="winning_continuation_line_count",
        selected_action=selected,
        line_universe=universe,
    )

    def capture(bindings):
        digits = bindings.continuation_line_digits
        assert bindings.continuation_line_universe == universe
        return {
            "PRED-COUNT": SimpleNamespace(
                literal=bindings.model.new_constant(1),
                extract=lambda solver: {
                    "count": str(sum(
                        int(solver.value(digit)) * (1_000_000**limb)
                        for limb, digit in enumerate(digits)
                    ))
                },
            )
        }

    monkeypatch.setattr(
        OrToolsScenarioGraphProofV1,
        "_solve_count",
        staticmethod(lambda *_args, **_kwargs: pytest.fail("native count used a second solve")),
    )
    proof = OrToolsScenarioGraphProofV1().prove(
        native_graph,
        request,
        native_line_count=True,
        tactic_configure=capture,
    )
    assert proof.solver_status == "optimal"
    assert proof.native_qualification["PRED-COUNT"]["count"] == str(expected)
    assert proof.continuation_lines.exact_count == str(expected)


def _power_of_two_graph(exponent: int = 128) -> ProofGraphV1:
    nodes = [
        ProofGraphNodeV1.decision(
            "root", "north_south", (_edge(1, "ROOT", "layer-0"),)
        )
    ]
    for index in range(exponent):
        child = f"layer-{index + 1}" if index + 1 < exponent else "win"
        nodes.append(
            ProofGraphNodeV1.decision(
                f"layer-{index}",
                "north_south" if index % 2 == 0 else "west_east",
                (
                    _edge(2, f"L-{index}", child),
                    _edge(3, f"R-{index}", child),
                ),
            )
        )
    nodes.append(ProofGraphNodeV1.leaf("win", 3))
    return ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=nodes,
    )


def test_native_line_count_keeps_large_action_edge_multiplicity() -> None:
    graph = _with_native_count_context(_power_of_two_graph(64))
    request = _request(
        claim="winning_continuation_line_count",
        selected_action="ROOT",
        line_universe="all_legal",
    )

    def capture(bindings):
        digits = bindings.continuation_line_digits
        return {
            "PRED-COUNT": SimpleNamespace(
                literal=bindings.model.new_constant(1),
                extract=lambda solver: {
                    "count": str(sum(
                        int(solver.value(digit)) * (1_000_000**limb)
                        for limb, digit in enumerate(digits)
                    ))
                },
            )
        }

    proof = OrToolsScenarioGraphProofV1().prove(
        graph, request, native_line_count=True, tactic_configure=capture,
    )
    assert proof.solver_status == "optimal"
    assert proof.native_qualification["PRED-COUNT"]["count"] == str(2**64)
    assert proof.continuation_lines.exact_count == str(2**64)


@pytest.mark.parametrize(
    ("selected", "configure"),
    [("A", None), (None, lambda _bindings: {})],
)
def test_native_line_count_refuses_missing_same_solve_scope(selected, configure) -> None:
    with pytest.raises(ValueError, match="selected action and same-solve bindings"):
        OrToolsScenarioGraphProofV1().prove(
            _graph(),
            _request(
                claim="winning_continuation_line_count",
                selected_action=selected,
                line_universe="all_legal",
            ),
            native_line_count=True,
            tactic_configure=configure,
        )


def test_limb_encoding_counts_more_than_int64_exactly() -> None:
    proof = OrToolsScenarioGraphProofV1().prove(
        _power_of_two_graph(),
        _request(
            claim="winning_continuation_line_count",
            selected_action="ROOT",
            line_universe="all_legal",
        ),
    )

    assert proof.continuation_lines is not None
    assert proof.continuation_lines.complete is True
    assert proof.continuation_lines.exact_count == str(2**128)
    assert proof.continuation_lines.limb_count == 7


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("universe", ("all_legal", "optimal_strategy"))
def test_limb_tier_refusal_preserves_safe_line_bounds(native, universe) -> None:
    graph = _power_of_two_graph()
    if native:
        graph = _with_native_count_context(graph)

    def unavailable_bindings(_bindings):
        pytest.fail("exhausted native count must not qualify a predicate")

    proof = OrToolsScenarioGraphProofV1().prove(
        graph,
        _request(
            claim="winning_continuation_line_count",
            selected_action="ROOT",
            line_universe=universe,
        ),
        maximum_line_count_limbs=6,
        **({"native_line_count": True, "tactic_configure": unavailable_bindings} if native else {}),
    )

    assert proof.completion_status == "proof_incomplete"
    assert proof.claim_satisfied is False
    assert proof.continuation_lines is not None
    assert proof.continuation_lines.complete is False
    assert proof.continuation_lines.exact_count is None
    assert proof.continuation_lines.lower_bound == "0"
    assert proof.continuation_lines.upper_bound == str(2**128)
    assert proof.continuation_lines.limb_count == 7
    assert proof.continuation_lines.limb_limit == 6
    assert proof.continuation_lines.exhaustion_reason == "maximum_line_count_limbs"
    assert proof.native_qualification is None
    if native:
        from baloot_platform.research.ortools_research_executor import _proof_receipt_evidence_bindings_v3

        assert proof.winning_root_action_count is None
        assert proof.winning_root_action_lower_bound == 0
        assert proof.winning_root_action_upper_bound == 1
        assert proof.model_document()["solvePerformed"] is False
        assert _proof_receipt_evidence_bindings_v3(proof, _request(
            claim="winning_continuation_line_count", selected_action="ROOT", line_universe=universe,
        )) == ()


def test_incomplete_count_solve_preserves_safe_line_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from baloot_platform.research.ortools_research_executor import _proof_receipt_evidence_bindings_v3

    class FakeSolver:
        @staticmethod
        def status_name(_status) -> str:
            return "FEASIBLE"

    monkeypatch.setattr(
        OrToolsScenarioGraphProofV1,
        "_solve_count",
        staticmethod(lambda _model, maximum_seconds: (FakeSolver(), cp_model.FEASIBLE)),
    )
    request = _request(
        claim="winning_continuation_line_count",
        selected_action="ROOT",
        line_universe="all_legal",
    )
    proof = OrToolsScenarioGraphProofV1().prove(_power_of_two_graph(8), request)

    assert proof.completion_status == "proof_incomplete"
    assert proof.solver_status == "optimal"
    assert proof.root_actions and proof.winning_root_action_count is not None
    bindings = _proof_receipt_evidence_bindings_v3(proof, request)
    assert [binding["predicateId"] for binding in bindings] == ["PROOF-CLAIM"]
    assert bindings[0]["sourceEvidenceHash"] == proof.sha256
    assert proof.continuation_lines is not None
    assert proof.continuation_lines.lower_bound == "0"
    assert proof.continuation_lines.upper_bound == str(2**8)
    assert proof.continuation_lines.exhaustion_reason == "line_count_incomplete"


def _policy_difference_graph() -> ProofGraphV1:
    return ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                "north_south",
                (_edge(1, "ROOT", "reply"),),
                actor_seat="south",
            ),
            ProofGraphNodeV1.decision(
                "reply",
                "west_east",
                (_edge(4, "REFUTE", "loss"), _edge(9, "ALLOW", "win")),
                actor_seat="west",
            ),
            ProofGraphNodeV1.leaf("loss", -1),
            ProofGraphNodeV1.leaf("win", 5),
        ),
    )


def test_policy_conditioned_truth_can_differ_from_universal_exact_truth() -> None:
    graph = _policy_difference_graph()
    universal = OrToolsScenarioGraphProofV1().prove(
        graph,
        _request(
            claim="outcome_reachable",
            selected_action="ROOT",
        ),
    )
    policy = OrToolsScenarioGraphProofV1().prove(
        graph,
        _request(
            claim="outcome_reachable",
            selected_action="ROOT",
            mode="policy_conditioned",
            policies=_policies(west_action=9),
        ),
    )

    assert universal.completion_status == "proof_failed"
    assert universal.root_actions[0].value == -1
    assert policy.completion_status == "policy_complete"
    assert policy.claim_satisfied is True
    assert policy.root_actions[0].value == 5
    assert policy.principal_variation == ("ROOT", "ALLOW")
    identities = policy.to_dict()["policyIdentities"]
    assert identities["west"]["policyId"] == "fixed_action_map_v1"
    assert identities["west"]["version"] == "1.0.0"
    assert identities["west"]["seed"] == 7
    assert identities["west"]["profileHash"] == _policies(west_action=9)["west"][
        "profileHash"
    ]


def test_policy_templates_materialize_actor_safe_fixed_action_maps() -> None:
    graph = ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                "north_south",
                (_edge(1, "ROOT", "west-a"),),
                actor_seat="south",
            ),
            ProofGraphNodeV1.decision(
                "west-a",
                "west_east",
                (_edge(4, "LOW", "west-b"), _edge(9, "HIGH", "loss")),
                actor_seat="west",
            ),
            ProofGraphNodeV1.decision(
                "west-b",
                "west_east",
                (_edge(4, "LOW", "win"), _edge(9, "HIGH", "loss")),
                actor_seat="west",
            ),
            ProofGraphNodeV1.leaf("loss", -1),
            ProofGraphNodeV1.leaf("win", 1),
        ),
    )
    request = _request(
        claim="outcome_reachable",
        selected_action="ROOT",
        mode="policy_conditioned",
        policies=_policy_templates(),
    )

    first = OrToolsScenarioGraphProofV1().prove(graph, request)
    second = OrToolsScenarioGraphProofV1().prove(graph, request)

    assert first.completion_status in {"policy_complete", "proof_failed"}
    assert first.sha256 == second.sha256
    west = next(item for item in first.policies if item.seat == "west")
    assert len(west.choices) == 2
    assert len({item.action_id for item in west.choices}) == 1
    assert west.profile_hash == _policy_templates()["west"]["profileHash"]
    assert west.materialized_hash != west.profile_hash
    assert first.to_dict()["policyIdentities"]["west"]["materializedHash"] == (
        west.materialized_hash
    )


def test_policy_template_rejects_runtime_bot_identity() -> None:
    policies = _policy_templates()
    policies["west"] = {
        "schemaVersion": 1,
        "agentId": "heuristic_bot_v1",
        "version": "1.0.0",
        "seed": 7,
        "profileHash": "0" * 64,
    }
    with pytest.raises(ValueError, match="fields are invalid"):
        OrToolsScenarioGraphProofV1().prove(
            _policy_difference_graph(),
            _request(
                claim="outcome_reachable",
                selected_action="ROOT",
                mode="policy_conditioned",
                policies=policies,
            ),
        )


def test_policy_conditioned_line_count_is_the_fixed_continuation() -> None:
    proof = OrToolsScenarioGraphProofV1().prove(
        _policy_difference_graph(),
        _request(
            claim="winning_continuation_line_count",
            selected_action="ROOT",
            mode="policy_conditioned",
            line_universe="policy_conditioned",
            policies=_policies(west_action=9),
        ),
    )

    assert proof.completion_status == "policy_complete"
    assert proof.continuation_lines is not None
    assert proof.continuation_lines.complete is True
    assert proof.continuation_lines.exact_count == "1"


@pytest.mark.parametrize(("west_action", "expected"), [(9, "1"), (4, "0")])
def test_native_policy_line_count_uses_selected_root_win_variable(
    west_action, expected
) -> None:
    graph = _with_native_count_context(_policy_difference_graph())
    request = _request(
        claim="winning_continuation_line_count",
        selected_action="ROOT",
        mode="policy_conditioned",
        line_universe="policy_conditioned",
        policies=_policies(west_action=west_action),
    )

    def capture(bindings):
        return {
            "PRED-COUNT": SimpleNamespace(
                literal=bindings.model.new_constant(1),
                extract=lambda solver: {
                    "count": str(solver.value(bindings.continuation_line_digits[0]))
                },
            )
        }

    proof = OrToolsScenarioGraphProofV1().prove(
        graph, request, native_line_count=True, tactic_configure=capture,
    )
    assert proof.solver_status == "optimal"
    assert proof.native_qualification["PRED-COUNT"]["count"] == expected
    assert proof.continuation_lines.exact_count == expected


@pytest.mark.parametrize(
    ("claim", "west_action", "line_universe", "winning_count"),
    (
        ("outcome_reachable", 9, None, 1),
        ("outcome_forced", 9, None, 1),
        ("winning_root_action_count", 9, None, 1),
        ("winning_continuation_line_count", 9, "policy_conditioned", 1),
        ("selected_action_is_unique_winner", 9, None, 1),
        ("target_team_has_no_salvage", 4, None, 0),
    ),
)
def test_policy_conditioned_supports_every_graph_claim(
    claim: str,
    west_action: int,
    line_universe: str | None,
    winning_count: int,
) -> None:
    proof = OrToolsScenarioGraphProofV1().prove(
        _policy_difference_graph(),
        _request(
            claim=claim,
            selected_action="ROOT",
            mode="policy_conditioned",
            line_universe=line_universe,
            policies=_policies(west_action=west_action),
        ),
    )

    assert proof.completion_status == "policy_complete"
    assert proof.claim_satisfied is True
    assert proof.winning_root_action_count == winning_count


def test_policy_conditioned_mapping_fails_closed() -> None:
    missing = _policies(west_action=9)
    missing["west"] = _policy("west", [])
    with pytest.raises(ValueError, match="select every non-root decision"):
        OrToolsScenarioGraphProofV1().prove(
            _policy_difference_graph(),
            _request(
                claim="outcome_reachable",
                selected_action="ROOT",
                mode="policy_conditioned",
                policies=missing,
            ),
        )

    illegal = _policies(west_action=99)
    with pytest.raises(ValueError, match="not legal"):
        OrToolsScenarioGraphProofV1().prove(
            _policy_difference_graph(),
            _request(
                claim="outcome_reachable",
                selected_action="ROOT",
                mode="policy_conditioned",
                policies=illegal,
            ),
        )


def test_policy_conditioned_feasible_status_remains_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSolver:
        @staticmethod
        def status_name(_status) -> str:
            return "FEASIBLE"

    monkeypatch.setattr(
        OrToolsScenarioGraphProofV1,
        "_solve",
        staticmethod(lambda _model, maximum_seconds: (FakeSolver(), cp_model.FEASIBLE)),
    )
    proof = OrToolsScenarioGraphProofV1().prove(
        _policy_difference_graph(),
        _request(
            claim="outcome_reachable",
            selected_action="ROOT",
            mode="policy_conditioned",
            policies=_policies(west_action=9),
        ),
    )

    assert proof.completion_status == "proof_incomplete"
    assert proof.claim_satisfied is False
    assert proof.root_actions == ()
    assert proof.winning_root_action_count is None
    assert proof.winning_root_action_lower_bound == 0
    assert proof.winning_root_action_upper_bound == 1
    assert proof.to_dict()["policyIdentities"]["west"]["seed"] == 7


def test_new_proof_module_has_no_engine_bot_or_legacy_proof_imports() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "baloot_platform"
        / "research"
        / "ortools_proof.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    forbidden = ("core.engine", "core.bots", "proof_service", "best_response_proof")
    assert not any(any(item in name for item in forbidden) for name in imports)
