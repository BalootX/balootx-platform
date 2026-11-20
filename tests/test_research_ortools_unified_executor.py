from __future__ import annotations

import copy
import json
import itertools
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import yaml
from deterministic_work import set_monotonic
from ortools.sat.python import cp_model

import baloot_platform.research.ortools_research_executor as executor_module

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.enums import Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import (
    OrToolsExactStateNormalizerV1,
    ProjectDecisionNotStrategicError,
)
from baloot_platform.research.ortools_play_tail import (
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
)
from baloot_platform.research.ortools_proof import (
    OrToolsGraphProofOutcomeV1,
    OrToolsScenarioGraphProofV1,
    ProofGraphEdgeV1,
    ProofGraphNodeV1,
    ProofGraphV1,
)
from baloot_platform.research.ortools_research_executor import (
    DefaultStageContextCompilerV1,
    GenericCompleteTraceCompilerV1,
    resolve_quality_play_continuation_v1,
    compile_observed_decision_v1,
    OrToolsCandidateAttemptV1,
    OrToolsNeutralCandidateV1,
    OrToolsResearchCompilerV1,
    OrToolsResearchArtifactPreimagesV1,
    OrToolsResearchCampaignResultV1,
    OrToolsResearchStageReceiptV1,
    OrToolsStageContextBindingV1,
    OrToolsUnifiedResearchExecutorV1,
    _campaign_incomplete_compilation_summary_v1,
    _campaign_variant_decision_cuts_v1,
    _campaign_variant_from_execution_v1,
    _proof_root_variables,
    _validate_play_policy_continuation_binding_v1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    ScenarioProofBudgetV1,
    ScenarioPlayingQualityPolicyV1,
    ScenarioQualityExecutionBudgetV1,
    SCENARIO_DECISION_KINDS,
    canonical_sha256,
)
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.research.tail_first_stage_planner import TailFirstStagePlannerV1


class _AllowanceReached(RuntimeError):
    pass


def _assert_parent_allowance(*args, maximum_seconds, **kwargs):
    assert maximum_seconds == 0.1
    raise _AllowanceReached


def test_observed_history_allowance_cannot_round_above_its_parent(rules, monkeypatch):
    from baloot_platform.research import ortools_nonplay_graphs as graphs

    clock = SimpleNamespace(monotonic=lambda: 1e9)
    monkeypatch.setattr(executor_module, "time", clock)
    monkeypatch.setattr(graphs, "time", clock)
    monkeypatch.setattr(executor_module, "validate_proof_boundary_scope_v1", lambda *_: None)
    monkeypatch.setattr(graphs.OrToolsEscalationStageV1, "solve", _assert_parent_allowance)
    normalized = SimpleNamespace(
        recorded_escalation_history=lambda: None, next_actor="south",
        state={"declarer": "south", "contract": "hokum"},
    )
    prefix = SimpleNamespace(escalation_actions=(), match_score=(0, 0))
    with pytest.raises(_AllowanceReached):
        compile_observed_decision_v1(
            rules, normalized, prefix, {"kind": "project"}, None,
            maximum_nodes=2, maximum_depth=1, maximum_worlds=1, maximum_seconds=0.1,
            native_cp_sat=True, private_recall={},
        )


@pytest.mark.parametrize("allowance,error,match", [
    pytest.param(0.1, _AllowanceReached, None, id="capped-query"),
    pytest.param(math.nextafter(0.1, math.inf), ValueError, "runtime allowance exceeds the declared cap", id="over-cap"),
])
def test_native_depth_query_allowance_cannot_round_above_its_parent(rules, monkeypatch, allowance, error, match):
    from baloot_platform.research import ortools_information_set_compiler as information

    monkeypatch.setattr(information, "time", SimpleNamespace(monotonic=lambda: 1e9))
    monkeypatch.setattr(information, "OrToolsNormalizedExactStateV1", SimpleNamespace)
    model = SimpleNamespace(
        model=SimpleNamespace(proto="round"), _model_hash=lambda **_: "model",
        find_native_continuation_depth_above=_assert_parent_allowance,
    )
    compiler = SimpleNamespace(rules=rules, _native_post_auction_root_model=lambda *args, **kwargs: model)
    exact = SimpleNamespace(next_actor="south", decision_kind="play", legal_actions=[{"code": "7S"}])
    with pytest.raises(error, match=match):
        information.OrToolsActorInformationSetCompilerV1._solve_native_post_auction_policy_bounds(
            compiler, exact, None, Team.NORTH_SOUTH, selected_root_action="7S",
            opportunity_order=(), escalation_events=(), maximum_iterations=1, maximum_queries=1,
            maximum_observations=1, maximum_model_bytes=100, maximum_conflicts=1, maximum_depth=1,
            maximum_seconds=0.1, runtime_maximum_seconds=allowance,
        )


@pytest.fixture
def deterministic_native_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    original_solve = cp_model.CpSolver.solve

    def solve(solver, *args, **kwargs):
        # Meter real single-worker solver work, independently of runner speed.
        assert solver.parameters.num_search_workers == 1
        solver.parameters.clear_max_time_in_seconds()
        solver.parameters.max_deterministic_time = 10.0
        status = original_solve(solver, *args, **kwargs)
        assert solver.deterministic_time <= solver.parameters.max_deterministic_time / 2, (
            f"CP-SAT deterministic time {solver.deterministic_time:.6f} exceeds 50% "
            f"of test limit {solver.parameters.max_deterministic_time}; investigate solver/model drift"
        )
        return status

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    return clock


def test_campaign_progress_references_incomplete_evidence_without_copying_payloads():
    full = {
        "stage": "playingQuality",
        "sha256": "a" * 64,
        "evidenceBytes": 12_000_000,
        "qualityFailureCauseStage": "hidden_world_enumeration",
        "qualityFailureQueryCount": 0,
        "qualityFailureQueryPhase": "path_query",
        "qualityFailurePathPhase": "solver",
        "recordedTrace": {"playActions": ["S7"] * 1_000},
        "qualityEvidence": {"decisions": [{"value": 1}] * 1_000},
    }

    summary = _campaign_incomplete_compilation_summary_v1(full)

    assert summary == {
        "stage": "playingQuality",
        "sha256": "a" * 64,
        "evidenceBytes": 12_000_000,
        "qualityFailureCauseStage": "hidden_world_enumeration",
        "qualityFailureQueryCount": 0,
        "qualityFailureQueryPhase": "path_query",
        "qualityFailurePathPhase": "solver",
    }


@pytest.mark.real_clock
@pytest.mark.parametrize("recorded_action", ["S7", "SA"])
def test_native_partial_prefix_audits_then_generates_without_repair(root, rules, recorded_action):
    prefix, request = _quality_continuation_input(root, "bounded")
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    state = normalizer.normalize(prefix, decision_point={"kind": "play"})
    submitted = normalizer.append_normalized_action(prefix, state, recorded_action)
    result = resolve_quality_play_continuation_v1(rules, prefix, request,
        recorded_trace=submitted, continue_recorded_prefix=True,
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1, maximum_seconds=60,
        maximum_decision_seconds=20, maximum_evidence_bytes=20_000_000)
    assert result["status"] == ("complete" if recorded_action == "S7" else "quality_failed"), result.get("partialReceipt")
    assert result["operation"] == "audit_then_generate"
    assert result["submittedPrefix"] == submitted.to_dict()
    assert result["finalPrefix"]["playActions"][:28] == submitted.to_dict()["playActions"]
    assert len(result["finalPrefix"]["playActions"]) == 32
    assert result["decisions"][0]["action"] == recorded_action
    assert result["decisions"][0]["selectionReason"] == "recorded_action"
    assert result["decisions"][0]["nextPrefixHash"] == submitted.sha256
    if recorded_action == "SA":
        assert result["partialReceipt"] == {
            "stage": "action_analysis",
            "reason": "recorded_action_not_optimal",
            "proofArtifactHash": result["decisions"][0]["proofArtifactHash"],
        }
    assert all(row["selectionReason"] == "native_actor_information_set_best_value"
               for row in result["decisions"][1:])
    stopped = resolve_quality_play_continuation_v1(rules, prefix, request,
        recorded_trace=submitted, continue_recorded_prefix=True, cancel_check=lambda: True,
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1, maximum_seconds=60,
        maximum_decision_seconds=20, maximum_evidence_bytes=20_000_000)
    assert stopped["status"] == "incomplete"
    assert stopped["submittedPrefix"] == submitted.to_dict()
    assert stopped["finalPrefix"] == prefix.to_dict()
    assert stopped["decisions"] == []
    changed = copy.deepcopy(result)
    changed["recordedPrefixHash"] = "0" * 64
    changed["sha256"] = canonical_sha256({key: value for key, value in changed.items()
        if key not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="submitted prefix ownership"):
        executor_module.validate_native_decision_quality_evidence_v1(changed,
            source_prefix=prefix, rules_hash=rules.sha256, request=request)


def test_historical_round_fails_current_eight_world_quality(root):
    raw = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text(encoding="utf-8"))["exactTracePrefix"]
    current = json.loads((root / "tests/fixtures/research-native-observer-budgeted-round-v3.json").read_text(encoding="utf-8"))
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    assert rules.sha256 == current["rulesHash"]
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    policy = ScenarioPlayingQualityPolicyV1.build(current["playingQuality"])
    profile = settings["proofBudgets"]["profiles"][policy.decisions["play"]["budgetProfile"]]
    recorded = ExactTracePrefixV1.build({**raw, "playActions": raw["playActions"][:17]})
    prefix = ExactTracePrefixV1.build({**raw, "playActions": raw["playActions"][:16]})

    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, recorded_trace=recorded, continue_recorded_prefix=True,
        stop_on_recorded_quality_failure=True, execution_budget=execution,
        maximum_nodes=profile["maximumNodes"], maximum_depth=profile["maximumDepth"],
        maximum_worlds=min(profile["maximumHiddenWorlds"], execution.maximum_hidden_worlds_per_decision),
        maximum_seconds=profile["maximumWallSeconds"], maximum_decision_seconds=profile["maximumWallSeconds"],
        maximum_evidence_bytes=execution.maximum_evidence_bytes,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"],
    )

    assert result["status"] == "quality_failed", result["partialReceipt"]
    assert result["partialReceipt"]["reason"] == "recorded_action_not_optimal"
    assert result["finalPrefix"] == recorded.to_dict()
    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    assert (decision["playIndex"], decision["actor"], decision["action"]) == (16, "west", "HJ")
    assert decision["actualActionOptimal"] is False and decision["regret"] == 38


def test_search_quality_stops_after_first_proven_inferior_recorded_action(root, rules):
    prefix, request = _quality_continuation_input(root, "bounded")
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    submitted = normalizer.append_normalized_action(
        prefix,
        normalizer.normalize(prefix, decision_point={"kind": "play"}),
        "SA",
    )

    result = resolve_quality_play_continuation_v1(
        rules,
        prefix,
        request,
        recorded_trace=submitted,
        continue_recorded_prefix=True,
        stop_on_recorded_quality_failure=True,
        maximum_nodes=100_000,
        maximum_depth=8,
        maximum_worlds=1,
        maximum_seconds=60,
        maximum_decision_seconds=20,
        maximum_evidence_bytes=20_000_000,
    )

    assert result["status"] == "quality_failed"
    assert result["partialReceipt"]["reason"] == "recorded_action_not_optimal"
    assert result["finalPrefix"] == submitted.to_dict()
    assert len(result["decisions"]) == len(result["artifacts"]) == 1
    assert result["decisions"][0]["actualActionOptimal"] is False


@pytest.mark.parametrize("asserted_phase", [None, "bidding", "playing"])
@pytest.mark.parametrize("visibility", ["actor_information_set", "complete_state"])
def test_native_partial_auction_preserves_bid_then_generates(root, rules, asserted_phase, visibility):
    source, request = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"}] * 6, escalationActions=[],
        projectActions=[], projectDeclarations=[], projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    submitted = ExactTracePrefixV1.build({**raw, "auctionActions": [{"action": "pass"}] * 7,
        "stateAssertions": {"phase": asserted_phase} if asserted_phase else {}})
    value = request.to_dict()
    value.pop("sha256")
    value.update(horizon={"unit": "plies", "value": 1}, boundaryEvaluator="current_raw_point_margin_v1",
                 visibility=visibility)
    request = ScenarioProofRequestV1.build(value)
    result = executor_module.resolve_quality_auction_continuation_v1(rules, prefix, request,
        recorded_trace=submitted, continue_recorded_prefix=True,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=30,
        maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000)
    if asserted_phase == "playing":
        assert result["status"] == "model_error"
        assert result["partialReceipt"]["stage"] == "selected_action_normalization"
        assert result["finalPrefix"] == prefix.to_dict()
        assert result["decisions"] == []
        return
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["submittedPrefix"] == submitted.to_dict()
    assert result["finalPrefix"]["auctionActions"][:7] == submitted.to_dict()["auctionActions"]
    assert len(result["decisions"]) == 2
    assert result["decisions"][0]["selectionReason"] == "recorded_action"
    assert result["decisions"][0]["nextPrefixHash"] == submitted.sha256
    assert result["decisions"][1]["selectionReason"] == (
        "native_complete_state_best_value" if visibility == "complete_state"
        else "native_actor_information_set_best_value"
    )
    executor_module.validate_native_decision_quality_evidence_v1(result,
        source_prefix=prefix, rules_hash=rules.sha256, request=request)


def test_quality_candidate_requires_native_backend_and_evidence(root, rules):
    legacy = OrToolsUnifiedResearchExecutorV1(
        rules, bundle_revision_id="CBR-QUALITY-TEST", bundle_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="require native V3"):
        legacy.candidate_from_quality_trace({}, maximum_seconds=1)
    native = OrToolsUnifiedResearchExecutorV1(
        RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"),
        bundle_revision_id="CBR-QUALITY-TEST", bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    )
    with pytest.raises(ValueError, match="requires immutable quality evidence"):
        native.candidate_from_quality_trace({}, maximum_seconds=1)


@pytest.mark.real_clock
@pytest.mark.parametrize("entry", ("candidate", "incomplete"))
def test_quality_readback_forwards_caller_owned_search_limits(rules, monkeypatch, entry):
    from baloot_platform.research import result_authority
    profiles = {"offline_heavyweight_v1": {"maximumConflicts": 12345}}
    policy, execution, proof_profiles = object(), object(), object()
    trace = {"playingQualityEvidence": {}}

    def validate(actual, **kwargs):
        if entry != "incomplete":
            assert actual is trace
        assert kwargs["rules"] is rules
        assert kwargs["required_search_profiles"] is profiles
        assert 0 < kwargs["maximum_seconds"] <= 1
        assert kwargs["cancel_check"] is None
        if entry == "candidate":
            assert kwargs["required_policy"] is policy
            assert kwargs["required_execution"] is execution
            assert kwargs["required_proof_profiles"] is proof_profiles
            assert kwargs["require_family_opening"] is True
        raise RuntimeError("verified readback boundary")

    monkeypatch.setattr(executor_module, "validate_trace_quality_evidence_v1", validate)
    monkeypatch.setattr(result_authority, "_validate_trace_quality_evidence_v1", validate)
    with pytest.raises(RuntimeError, match="verified readback boundary"):
        if entry == "candidate":
            owner = SimpleNamespace(backend_id="ortools_cp_sat_v3", rules=rules,
                quality_search_profiles=profiles, quality_policy=policy,
                quality_execution=execution, quality_proof_profiles=proof_profiles)
            OrToolsUnifiedResearchExecutorV1.candidate_from_quality_trace(owner, trace, maximum_seconds=1)
        else:
            policy = SimpleNamespace(decisions={"bid": {"budgetProfile": "test"}})
            execution = SimpleNamespace(maximum_hidden_worlds_per_decision=1,
                featured_play_native_proof=False, escalation_opportunity_order=(),
                maximum_wall_seconds=60, maximum_escalation_decisions=1,
                maximum_evidence_bytes=1000)
            budget = SimpleNamespace(maximum_nodes=1, maximum_depth=1,
                maximum_hidden_worlds=1, maximum_wall_seconds=60)
            monkeypatch.setattr(executor_module, "_quality_budget_snapshots",
                lambda *args, execution_budget=None: ({"test": budget}, {}))
            def incomplete(_rules, source, *args, **kwargs):
                return {"status": "incomplete", "finalPrefix": source.to_dict(),
                    "executionReceipt": {"auction": {"elapsedSeconds": 0}}}
            monkeypatch.setattr(executor_module, "resolve_quality_round_continuation_v1", incomplete)
            OrToolsUnifiedResearchExecutorV1._resolve_quality_candidate(
                SimpleNamespace(rules=rules, backend_id="ortools_cp_sat_v3",
                    quality_policy=policy, quality_execution=execution,
                    quality_proof_profiles={}, quality_search_profiles=profiles, bid_quality_cache=None),
                SimpleNamespace(trace={"deckOrder": list(range(32)), "dealer": "east"},
                    origin="search"), maximum_seconds=1)


def test_play_continuation_rejects_audited_round_before_semantic_work(rules, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("audited continuation must not be validated or normalized here")

    monkeypatch.setattr(executor_module, "validate_trace_quality_evidence_v1", forbidden)
    monkeypatch.setattr(executor_module.OrToolsExactStateNormalizerV1, "normalize", forbidden)
    with pytest.raises(executor_module._PolicyContinuationCompilerDisagreement,
                       match="audited full rounds cannot be rematerialized"):
        executor_module._materialize_play_policy_continuation_v1(
            rules=rules, candidate=SimpleNamespace(trace={"playingQualityEvidence": {}}),
            source_normalization=None, graph=None, proof=None, request=None,
            maximum_seconds=1, native_cp_sat=True,
        )


@pytest.mark.real_clock
@pytest.mark.parametrize("stop_stage", (None, "start", "validation", "normalization", "candidate"))
@pytest.mark.parametrize("cancelled", (False, True))
def test_quality_readback_shares_deadline_across_all_stages(rules, monkeypatch, stop_stage, cancelled):
    clock = [0.0]
    calls = []
    cancellation = [False]
    prefix = SimpleNamespace(play_actions=tuple(range(32)))
    candidate = object()

    def stage(name):
        calls.append(name)
        clock[0] += 1
        if stop_stage == name:
            if cancelled:
                cancellation[0] = True
            else:
                clock[0] = 10

    def normalize(self, value, **kwargs):
        assert value is prefix and kwargs["maximum_seconds"] == 9
        stage("normalization")
        return object()

    def build(*args, **kwargs):
        stage("candidate")
        return candidate

    monkeypatch.setattr(executor_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(executor_module, "validate_trace_quality_evidence_v1", lambda *a, **k: stage("validation"))
    monkeypatch.setattr(executor_module.ExactTracePrefixV1, "build", lambda value: prefix)
    monkeypatch.setattr(executor_module.OrToolsExactStateNormalizerV1, "normalize", normalize)
    owner = SimpleNamespace(backend_id="ortools_cp_sat_v3", rules=rules,
        quality_search_profiles=None, quality_policy=None, quality_execution=None,
        quality_proof_profiles=None, _candidate_from_terminal_quality_trace=build)
    trace = {"completeExactTracePrefix": {}, "playingQualityEvidence": {
        "receipt": {"status": "complete", "operation": "generate"}}}
    if stop_stage == "start":
        if cancelled:
            cancellation[0] = True
        else:
            ticks = iter((0.0, 10.0))
            monkeypatch.setattr(executor_module.time, "monotonic", lambda: next(ticks))
    def run():
        return OrToolsUnifiedResearchExecutorV1.candidate_from_quality_trace(owner, trace,
            maximum_seconds=10, cancel_check=lambda: cancellation[0])
    if stop_stage is None:
        assert run() is candidate
        assert calls == ["validation", "normalization", "candidate"]
    else:
        with pytest.raises(executor_module.InformationSetCompilationStopped) as caught:
            run()
        assert caught.value.reason == ("cancelled" if cancelled else "maximum_seconds")
        assert calls == [] if stop_stage == "start" else calls[-1] == stop_stage


def test_quality_readback_rejects_other_visibility_policy_before_normalization(root, rules, monkeypatch):
    source, _ = _quality_continuation_input(root, "bounded")
    raw = {"schemaVersion": 2, "mode": "exact", "visibility": "complete_state",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "offline_heavyweight_v1"}
            for kind in SCENARIO_DECISION_KINDS}}
    required = ScenarioPlayingQualityPolicyV1.build(raw)
    other = ScenarioPlayingQualityPolicyV1.build({**raw, "visibility": "actor_information_set"})
    owner = SimpleNamespace(backend_id="ortools_cp_sat_v3", rules=rules,
        quality_policy=required, quality_execution=None, quality_proof_profiles=None,
        quality_search_profiles=None)
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched quality policy reached normalization")
    monkeypatch.setattr(executor_module.OrToolsExactStateNormalizerV1, "normalize", forbidden)
    trace = {"playingQualityEvidence": {"sourcePrefix": source.to_dict(),
        "decisionPoint": {"kind": "play", "selector": {}}, "receipt": {"playingQuality": other.to_dict()}}}
    with pytest.raises(ValueError, match="differs from requested policy"):
        OrToolsUnifiedResearchExecutorV1.candidate_from_quality_trace(owner, trace, maximum_seconds=1)


@pytest.mark.real_clock
@pytest.mark.parametrize("seconds", (True, 0, -1, float("inf"), float("nan")))
def test_quality_readback_rejects_invalid_deadline_before_validation(rules, seconds):
    owner = SimpleNamespace(backend_id="ortools_cp_sat_v3", rules=rules)
    with pytest.raises(ValueError, match="deadline must be finite and positive"):
        OrToolsUnifiedResearchExecutorV1.candidate_from_quality_trace(owner, {}, maximum_seconds=seconds)


@pytest.mark.real_clock
@pytest.mark.parametrize("caller_seconds,expected", [(None, 30), (5, 5)])
def test_direct_quality_readback_defaults_only_to_frozen_execution_budget(monkeypatch, caller_seconds, expected):
    from baloot_platform.research import result_authority as authority
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 30, "maximumEvidenceBytes": 100000,
        "maximumEscalationDecisions": 16})
    original = authority._quality_readback_clock
    allowances = []
    def clock(seconds, cancel):
        allowances.append(seconds)
        return original(seconds, cancel)
    monkeypatch.setattr(authority, "_quality_readback_clock", clock)
    with pytest.raises(ValueError, match="kind is unsupported"):
        authority._validate_native_decision_quality_evidence_v1({}, source_prefix=None,
            rules_hash="a" * 64, request=None, execution_budget=execution,
            maximum_seconds=caller_seconds)
    assert allowances == [expected]


@pytest.mark.real_clock
@pytest.mark.parametrize("caller_seconds,expected", [(None, 30), (5, 5)])
def test_round_quality_readback_shares_frozen_default_across_stages(monkeypatch, caller_seconds, expected):
    from baloot_platform.research import result_authority as authority
    elapsed = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "matchScore": [0, 0], "auctionActions": [],
        "escalationActions": [], "projectDeclarations": [], "playActions": []})
    policy = SimpleNamespace(to_dict=lambda: {}, request_for=lambda *a: None)
    common = {"rulesHash": "a" * 64, "sourcePrefixHash": prefix.sha256,
        "status": "complete", "operation": "generate", "playingQuality": {},
        "finalPrefix": prefix.to_dict()}
    auction = {**common, "auctionOutcome": {"allPassed": False}}
    escalation = {**common}
    play = {**common}
    post = {**common, "kind": "native_quality_post_auction_continuation_v1",
        "scope": "post_auction_only", "escalation": escalation, "projectPlay": play}
    post["sha256"] = canonical_sha256(post)
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 30, "maximumEvidenceBytes": 100000,
        "maximumEscalationDecisions": 16})
    value = {**common, "kind": "native_quality_round_continuation_v1",
        "scope": "all_decisions", "status": "incomplete", "partialReceipt": {},
        "qualityExecution": execution.to_dict(), "auction": auction, "postAuction": post}
    value["sha256"] = canonical_sha256(value)
    allowances = []
    def validate(*a, **kwargs):
        allowances.append(kwargs["maximum_seconds"])
        elapsed[0] += 1
    monkeypatch.setattr(authority, "_validate_native_decision_quality_evidence_v1", validate)
    monkeypatch.setattr(authority, "validate_native_escalation_quality_evidence_v1", validate)
    authority._validate_native_round_quality_evidence_v1(value, source_prefix=prefix,
        rules_hash="a" * 64, policy=policy, decision_point={"kind": "play"},
        maximum_seconds=caller_seconds)
    assert allowances == [expected, expected - 1, expected - 2]


@pytest.mark.parametrize("field,attribute", [
    ("allowExactInputQuality", "allow_exact_input_quality"),
    ("completeStatePlayPolicyPair", "complete_state_play_policy_pair"),
    ("pruneOwnProjectPassWorlds", "prune_own_project_pass_worlds"),
    ("auditConstructedTrace", "audit_constructed_trace"),
])
def test_exact_input_quality_flag_is_strict_hashed_and_legacy_safe(field, attribute):
    raw = {"schemaVersion": 1, "maximumWallSeconds": 60,
           "maximumEvidenceBytes": 1000000, "maximumEscalationDecisions": 20}
    legacy = ScenarioQualityExecutionBudgetV1.build(raw)
    assert not getattr(legacy, attribute)
    assert field not in legacy.to_dict()
    enabled = ScenarioQualityExecutionBudgetV1.build({**raw, field: True})
    assert enabled.sha256 != legacy.sha256
    assert ScenarioQualityExecutionBudgetV1.build(enabled.to_dict()) == enabled
    assert ScenarioQualityExecutionBudgetV1.build({**raw, field: False}) == legacy
    for invalid in (1, "true", None, []):
        with pytest.raises(ValueError, match="must be boolean"):
            ScenarioQualityExecutionBudgetV1.build({**raw, field: invalid})
    with pytest.raises(ValueError, match="checksum"):
        ScenarioQualityExecutionBudgetV1.build({**enabled.to_dict(), field: False})


@pytest.mark.parametrize("enabled", (False, True))
def test_exact_input_quality_routes_only_when_frozen_flag_enabled(root, monkeypatch, enabled):
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    execution = ScenarioQualityExecutionBudgetV1.build({**profile["qualityExecution"], "allowExactInputQuality": enabled})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact", "visibility": "complete_state",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "offline_heavyweight_v1"}
                      for kind in SCENARIO_DECISION_KINDS}})
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["projectActions"].append({"actor": "east", "action": "pass"})
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": raw,
                                "recordedEscalationOpportunities": []},
        constraints=_leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {}})
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-EXACT-ROUTING-TEST", bundle_hash="a" * 64,
        quality_policy=policy, quality_execution=execution,
        quality_proof_profiles=profile["proofBudgets"]["profiles"])
    class AuditReached(Exception):
        pass
    def audit(self, candidate, **kwargs):
        assert self.quality_policy == policy
        assert kwargs["recorded_trace"] == scenario.origin.exact_trace_prefix
        assert kwargs["recorded_events"] == ()
        assert kwargs["continue_recorded_prefix"] is True
        assert kwargs["stop_on_recorded_quality_failure"] is True
        raise AuditReached()
    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "audit_quality_candidate", audit)
    with pytest.raises(AuditReached if enabled else ValueError):
        executor.execute(scenario, _proof(), maximum_seconds=30,
            maximum_proof_nodes=1000, maximum_proof_depth=64, maximum_hidden_worlds=1)


@pytest.mark.parametrize("exact_from,expected_steps", [(1, 0), (2, 4), (6, 20), (7, 23), (8, 23)])
def test_tail_search_policy_horizon_matches_quality_boundary(exact_from, expected_steps):
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "tricks", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS},
    })
    budget = ScenarioQualityExecutionBudgetV1.build({
        "schemaVersion": 1, "maximumWallSeconds": 60,
        "maximumEvidenceBytes": 1_000_000, "maximumEscalationDecisions": 20,
        "exactPlayFromTrick": exact_from,
        **({"earlyPlayPolicyId": "observer_safe_tactical_category_v1"}
           if exact_from > 1 else {}),
    })
    assert executor_module._tail_early_policy_steps_v1(23, policy, budget) == expected_steps
    assert executor_module._tail_early_policy_steps_v1(23, None, budget) == 0
    exact_policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "exact", "visibility": "complete_state",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "offline_heavyweight_v1",
        } for kind in SCENARIO_DECISION_KINDS},
    })
    assert executor_module._tail_early_policy_steps_v1(23, exact_policy, budget) == 0


@pytest.mark.parametrize("prune_pass_worlds", [False, True])
@pytest.mark.parametrize("featured_now", [False, True])
def test_native_from_first_play_skips_heuristic_policy(
    rules, monkeypatch, prune_pass_worlds, featured_now,
):
    class NativeReached(Exception):
        pass

    def native_play(compiler, *_args, **_kwargs):
        assert compiler.prune_own_project_pass_worlds is prune_pass_worlds
        raise NativeReached

    monkeypatch.setattr(
        executor_module.OrToolsActorInformationSetCompilerV1,
        "_prove_observed_play" if featured_now else "_compile_observed_play_selection", native_play,
    )
    monkeypatch.setattr(
        executor_module, "_observer_safe_play_policy_artifact_v1",
        lambda *_args, **_kwargs: pytest.fail("native-from-first must not use early policy"),
    )
    quality_policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "tricks", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS},
    })
    execution = ScenarioQualityExecutionBudgetV1.build({
        "schemaVersion": 1, "maximumWallSeconds": 60,
        "maximumEvidenceBytes": 1_000_000, "maximumEscalationDecisions": 20,
        "exactPlayFromTrick": 1,
        **({"pruneOwnProjectPassWorlds": True} if prune_pass_worlds else {}),
    })
    with pytest.raises(NativeReached):
        executor_module._select_quality_play_action_v1(
            rules, None,
            SimpleNamespace(decision_kind="play", legal_actions=({"code": "S7"}, {"code": "S8"}),
                            state={"currentTrick": {"index": 0}}),
            quality_policy.request_for("play", "north"), quality_policy=quality_policy, execution_budget=execution,
            budget=SimpleNamespace(native_observed=True), featured_now=featured_now,
            actual_action=None, remaining_seconds=lambda: 1.0,
            cancel_check=None, private_recall=None, maximum_worlds=1, maximum_depth=1,
        )


@pytest.mark.real_clock
@pytest.mark.parametrize("artifact_kind", ["native_complete_state_root_outcome_v1", "native_bounded_play_quality_model_v1"])
def test_search_quality_regenerates_from_constructed_deal_before_continuing(
    root, rules, complete_candidate, monkeypatch, artifact_kind,
):
    from baloot_platform.research.ortools_tail_first_candidate import (
        OrToolsTailFirstCandidateCompilerV1,
    )
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2

    profile = yaml.safe_load(
        (root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8")
    )
    backend = OrToolsScenarioResearchBackendV2(
        challenge_profile=profile,
        bundle_identity={
            "bundleRevisionId": "CBR-SEARCH-QUALITY-PREFIX",
            "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64,
        },
    )
    policy = ScenarioPlayingQualityPolicyV1.build(
        {
            "schemaVersion": 1,
            "mode": "budgeted",
            "decisions": {
                kind: {
                    "utility": "awarded_point_margin",
                    "budgetProfile": "late_game_exact_v1",
                    "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
                    "boundaryEvaluator": "current_raw_point_margin_v1",
                }
                for kind in SCENARIO_DECISION_KINDS
            },
        }
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"),
        backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-SEARCH-QUALITY-PREFIX",
        bundle_hash="a" * 64,
        quality_policy=policy,
        quality_execution=backend.authority.quality_execution,
        quality_proof_profiles=profile["scenarioResearchV4"]["proofBudgets"]["profiles"],
    )
    scenario = _matching_scenario(complete_candidate)

    verified = set()
    def capture_match_mode(*_args, **kwargs):
        assert kwargs["_verified_artifacts"] is verified
        assert kwargs["featured_selector_dynamic"] is True
        assert kwargs["runtime_maximum_seconds"] == 7
        raise RuntimeError("search match mode checked")

    with monkeypatch.context() as match_mode_patch:
        handoff_clock = iter((100.0, 103.0))
        match_mode_patch.setattr(executor_module.time, "perf_counter", lambda: next(handoff_clock, 103.0))
        match_mode_patch.setattr(
            executor_module,
            "resolve_quality_round_continuation_v1",
            capture_match_mode,
        )
        with pytest.raises(RuntimeError, match="search match mode checked"):
            executor.generate_quality_candidate(
                complete_candidate,
                scenario=scenario,
                maximum_seconds=10,
                _verified_artifacts=verified,
            )
    for audit_recorded in (False, True):
        with monkeypatch.context() as expired_patch:
            expired_clock = iter((100.0, 111.0))
            expired_patch.setattr(executor_module.time, "perf_counter", lambda: next(expired_clock, 111.0))
            expired_patch.setattr(executor_module, "resolve_quality_round_continuation_v1",
                lambda *args, **kwargs: pytest.fail("expired handoff must not launch quality work"))
            if audit_recorded:
                expired = executor.audit_quality_candidate(complete_candidate,
                    recorded_trace=executor_module._construction_auction_prefix_v1(complete_candidate),
                    recorded_events=(), maximum_seconds=10,
                    decision_point={"kind": "escalation", "selector": {}},
                    continue_recorded_prefix=True)
            else:
                expired = executor.generate_quality_candidate(complete_candidate,
                    scenario=scenario, maximum_seconds=10)
            assert expired.status == "incomplete"
            assert expired.partial_receipt["qualityFailureReason"] == "maximum_seconds"
            assert expired.reason == ("quality_recorded_audit_incomplete" if audit_recorded
                                      else "quality_generation_incomplete")
    calls = []
    construction_calls = []
    tail_rejections = []
    live = []
    admission_reuse = []
    original_result = executor._result
    def capture_admission(*args, **kwargs):
        assert kwargs["_quality_verified_artifacts"] is calls[-1][1]["_verified_artifacts"]
        admission_reuse.append(frozenset(kwargs["_quality_verified_artifacts"]))
        return original_result(*args, **kwargs)
    monkeypatch.setattr(executor, "_result", capture_admission)
    second_deck = tuple(reversed(complete_candidate.no_good_vector))
    second_candidate = replace(
        complete_candidate,
        trace={**complete_candidate.trace, "deckOrder": list(second_deck)},
        no_good_vector=second_deck,
    )

    def candidate(*args, **kwargs):
        construction_calls.append(args[2])
        return OrToolsCandidateAttemptV1(
            "feasible", complete_candidate if len(construction_calls) == 1 else second_candidate,
        )

    def generate(original, **kwargs):
        calls.append((original, kwargs))
        artifact = {"proofModel": {"kind": artifact_kind},
                    "testCandidate": original.sha256}
        kwargs["checkpoint_callback"]({"kind": "quality_proof_checkpoint_v1",
            "artifact": artifact, "artifactHash": canonical_sha256(artifact)})
        kwargs["checkpoint_callback"]({"kind": "quality_proof_checkpoint_v1",
            "artifact": artifact, "artifactHash": "0" * 64})
        kwargs["checkpoint_callback"]({"kind": "quality_decision_checkpoint_v1",
            "decision": {"action": "private-test-action", "value": 99}})
        return OrToolsCandidateAttemptV1(
            "not_found", None, reason="quality_generation_target_miss"
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("search quality must not audit the arbitrary construction line")

    monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__", candidate)
    monkeypatch.setattr(
        OrToolsTailFirstCandidateCompilerV1,
        "__call__",
        lambda *_args, **_kwargs: pytest.fail(
            "quality search must not build a competing tail-first play"
        ),
    )
    monkeypatch.setattr(
        OrToolsTailFirstCandidateCompilerV1,
        "_reject_quality_candidate",
        lambda _compiler, rejected, **_kwargs: tail_rejections.append(rejected),
    )
    monkeypatch.setattr(executor, "generate_quality_candidate", generate)
    monkeypatch.setattr(executor, "audit_quality_candidate", forbidden)

    result = executor.execute(
        scenario,
        _proof(target_team=complete_candidate.trace["score"]["roundWinner"]),
        maximum_candidates=2,
        maximum_seconds=10,
        maximum_proof_nodes=1_000,
        maximum_proof_depth=8,
        maximum_prefix_lift_attempts=1,
        progress_callback=live.append,
    )

    assert result.status == "semantic_postcondition_miss"
    assert admission_reuse == [frozenset({canonical_sha256({
        "proofModel": {"kind": artifact_kind},
        "testCandidate": second_candidate.sha256})})]
    safe_progress = [row for row in live if row.get("currentStage") == "playingQuality"]
    assert safe_progress and all("decision" not in row and "artifact" not in row
                                 and "action" not in row and "value" not in row for row in safe_progress)
    assert "_quality_verified_artifacts" not in result.execution_receipt
    import pickle
    transported = pickle.loads(pickle.dumps(result))
    assert transported._quality_verified_artifacts == admission_reuse[0]
    assert transported.to_dict() == result.to_dict()
    exported = result.to_dict()
    assert exported["sha256"] == canonical_sha256({key: value for key, value in exported.items() if key != "sha256"})
    exported["executionReceipt"]["stageTimings"]["total"] = -1
    assert result.execution_receipt["stageTimings"]["total"] >= 0
    assert "_quality_verified_artifacts" not in transported.to_dict()
    assert replace(transported)._quality_verified_artifacts == frozenset()
    assert len(calls) == 2
    assert tail_rejections == []
    assert result.rejection_hashes == (complete_candidate.sha256, second_candidate.sha256)
    assert construction_calls == [(), (complete_candidate.no_good_vector,)]
    assert result.execution_receipt["candidateAttemptReason"] == "quality_generation_target_miss"
    assert "incompleteCompilation" not in result.execution_receipt
    timings = result.execution_receipt["stageTimings"]
    assert timings["construction"] > 0
    assert timings["playingQuality"] > 0
    assert timings["total"] >= timings["construction"] + timings["playingQuality"]
    original, kwargs = calls[0]
    assert original is complete_candidate
    assert kwargs["maximum_seconds"] > 0
    assert kwargs["cancel_check"] is None
    assert kwargs["scenario"] is scenario


@pytest.mark.real_clock
@pytest.mark.parametrize("quality_mode, policy_pair, visibility", [
    ("budgeted", True, "complete_state"),
    ("budgeted", True, "actor_information_set"),
    ("exact", False, "complete_state"),
    ("exact", True, "complete_state"),
])
@pytest.mark.parametrize("audit_constructed", [False, True])
def test_tail_quality_miss_excludes_only_the_recorded_play(
    root, complete_candidate, monkeypatch, quality_mode, policy_pair, visibility, audit_constructed,
):
    from baloot_platform.research.ortools_tail_first_candidate import (
        OrToolsTailFirstCandidateCompilerV1,
    )

    scenario = _scenario(
        constraints={"op": "all", "children": [
            _leaf("PRED-TAIL-SUN", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-TAIL-ACE", "play.action_matches", {
                "trickIndex": 5, "position": 3,
                "selector": {"ranks": ["ace"]},
            }),
        ]},
        outcome=_leaf("PRED-TAIL-REPORT", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 5, "position": 3, "remainingCards": 3,
            "actingSeat": "west",
        }},
    )
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.planner == "tail_first"
    candidate = replace(complete_candidate, trace={
        **complete_candidate.trace, "tailFirstStagePlan": plan.to_dict(),
    })
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": quality_mode, "visibility": visibility, "decisions": {
            kind: {"utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
                   **({} if quality_mode == "exact" else {
                   "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
                   "boundaryEvaluator": profile["authoringDefaults"]["playingQuality"][
                       "boundaryEvaluatorByDecision"][kind]})}
            for kind in SCENARIO_DECISION_KINDS
        },
    })
    executor = OrToolsUnifiedResearchExecutorV1(
        RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"),
        backend_id="ortools_cp_sat_v3", bundle_revision_id="CBR-TAIL-QUALITY-NOGOOD",
        bundle_hash="a" * 64, quality_policy=policy,
        quality_execution=ScenarioQualityExecutionBudgetV1.build({
            **profile["qualityExecution"], "completeStatePlayPolicyPair": policy_pair,
            "auditConstructedTrace": audit_constructed}),
        quality_proof_profiles=profile["proofBudgets"]["profiles"],
    )
    seen_exclusions = []
    rejected_plays = []

    def construct(_compiler, _scenario, excluded, **_kwargs):
        seen_exclusions.append(tuple(excluded))
        return (OrToolsCandidateAttemptV1("feasible", candidate) if len(seen_exclusions) == 1
                else OrToolsCandidateAttemptV1("incomplete", None, "test_second_attempt"))

    partial = {"schemaVersion": 1, "stage": "playingQuality", "qualityStatus": "quality_failed"}
    monkeypatch.setattr(OrToolsTailFirstCandidateCompilerV1, "__call__", construct)
    monkeypatch.setattr(OrToolsTailFirstCandidateCompilerV1, "_reject_quality_candidate",
        lambda _compiler, rejected: rejected_plays.append(rejected.sha256))
    generate = not audit_constructed
    original_hash = candidate.sha256

    def quality(original, **kwargs):
        assert original.sha256 == original_hash
        if generate:
            assert kwargs["scenario"] is scenario
            assert "recorded_trace" not in kwargs
        else:
            assert kwargs["recorded_trace"] == executor_module._construction_complete_prefix_v1(original)
        return OrToolsCandidateAttemptV1("not_found", None,
            "quality_generation_target_miss" if generate else "quality_recorded_audit_target_miss",
            partial_receipt={**partial, "sha256": canonical_sha256(partial)})

    monkeypatch.setattr(executor, "generate_quality_candidate" if generate else "audit_quality_candidate", quality)
    monkeypatch.setattr(executor, "audit_quality_candidate" if generate else "generate_quality_candidate",
        lambda *_args, **_kwargs: pytest.fail("wrong tail-first quality authority"))

    result = executor.execute(
        scenario, _proof(target_team="west_east"), maximum_proof_nodes=1000,
        maximum_proof_depth=12, maximum_prefix_lift_attempts=1,
        maximum_candidates=2, maximum_seconds=30,
    )
    assert result.status == "incomplete"
    assert result.rejection_hashes == (candidate.sha256,)
    assert seen_exclusions == [(), ()]
    assert rejected_plays == [candidate.sha256]

    executor.execute(
        scenario, _proof(target_team="west_east"), maximum_proof_nodes=1000,
        maximum_proof_depth=12, maximum_prefix_lift_attempts=1,
        maximum_candidates=1, maximum_seconds=30,
        initial_quality_rejected_candidates=(candidate,),
    )
    assert seen_exclusions == [(), (), ()]
    assert rejected_plays == [candidate.sha256, candidate.sha256]

    source_rejections = []
    monkeypatch.setattr(OrToolsTailFirstCandidateCompilerV1, "_reject_quality_candidate",
        lambda _compiler, rejected, **kwargs: source_rejections.append((rejected.sha256, kwargs)))
    source = {"deckOrder": candidate.trace["deckOrder"], "dealer": candidate.trace["dealer"],
              "roundNumber": candidate.trace.get("roundNumber", 1),
              "matchScore": candidate.trace.get("matchScore", [0, 0])}
    partial.update(reason="decision_point_target_miss", candidateHash=candidate.sha256,
                   sourcePrefix=source)
    seen_exclusions.clear()
    executor.execute(
        scenario, _proof(target_team="west_east"), maximum_proof_nodes=1000,
        maximum_proof_depth=12, maximum_prefix_lift_attempts=1,
        maximum_candidates=2, maximum_seconds=30,
    )
    source_eligible = generate and quality_mode == "budgeted" and visibility == "actor_information_set"
    assert source_rejections == [(candidate.sha256,
        {"reject_initial_input": True} if source_eligible else {})]
    assert seen_exclusions == [(), ()]

    if source_eligible:
        for changed in (
            {"candidateHash": "0" * 64}, {"reason": "maximum_seconds"},
            {"sourcePrefix": {**source, "matchScore": [1, 0]}},
            {"sourcePrefix": {**source, "roundNumber": source["roundNumber"] + 1}},
            {"sourcePrefix": {**source, "deckOrder": list(reversed(source["deckOrder"]))}},
        ):
            original_partial = dict(partial)
            partial.update(changed)
            source_rejections.clear()
            seen_exclusions.clear()
            executor.execute(
                scenario, _proof(target_team="west_east"), maximum_proof_nodes=1000,
                maximum_proof_depth=12, maximum_prefix_lift_attempts=1,
                maximum_candidates=2, maximum_seconds=30,
            )
            assert source_rejections == [(candidate.sha256, {})]
            partial.clear()
            partial.update(original_partial)

    target_scenario = _scenario(
        constraints=scenario.constraints.to_dict(),
        outcome=scenario.outcome.to_dict(),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
            "actingSeat": complete_candidate.trace["play"]["plays"][24]["actor"],
        }},
    )
    target = len(executor_module._search_decision_prefix(
        complete_candidate, target_scenario, native_cp_sat=True,
    ).play_actions)
    target_actor = target_scenario.decision_point["selector"]["actingSeat"]
    other_actor = next(seat.name.lower() for seat in Seat
                       if seat.name.lower() != target_actor)

    constructed_first = complete_candidate.trace["play"]["plays"][0]
    different_first_action = LegalPlayActionV1(
        Card.parse(next(
            code for code in ("D7", "C7", "H7", "S7")
            if code != constructed_first["card"]
        )),
        False,
    ).code
    class ContinuedAfterDivergence(Exception):
        pass

    structural_candidate = replace(
        complete_candidate,
        trace={
            key: value
            for key, value in complete_candidate.trace.items()
            if key not in {"play", "score"}
        },
    )

    def explicit_index_quality(*args, **kwargs):
        assert kwargs["featured_play_index"] == 24
        kwargs["checkpoint_callback"]({
            "kind": "quality_decision_checkpoint_v1", "stage": "play",
            "decision": {"playIndex": 0, "actor": constructed_first["actor"],
                         "action": different_first_action},
        })
        raise ContinuedAfterDivergence

    with monkeypatch.context() as explicit_index_patch:
        explicit_index_patch.setattr(
            executor_module,
            "_search_decision_prefix",
            lambda *_args, **_kwargs: pytest.fail(
                "explicit play coordinates must not depend on construction play"
            ),
        )
        explicit_index_patch.setattr(
            executor_module,
            "resolve_quality_round_continuation_v1",
            explicit_index_quality,
        )
        for partial in (structural_candidate, replace(structural_candidate,
                trace={**structural_candidate.trace, "play": {}})):
            with pytest.raises(ContinuedAfterDivergence):
                OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
                    executor, partial, scenario=target_scenario, maximum_seconds=10,
                )

    def divergent_quality(*args, **kwargs):
        assert kwargs["featured_decision_point"] == target_scenario.decision_point
        kwargs["checkpoint_callback"]({
            "schemaVersion": 1,
            "kind": "quality_decision_checkpoint_v1",
            "stage": "play",
            "decision": {
                "playIndex": 0,
                "actor": constructed_first["actor"],
                "action": different_first_action,
            },
        })
        raise ContinuedAfterDivergence

    with monkeypatch.context() as divergence_patch:
        divergence_patch.setattr(
            executor_module, "resolve_quality_round_continuation_v1", divergent_quality,
        )
        with pytest.raises(ContinuedAfterDivergence):
            OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
                executor, complete_candidate, scenario=target_scenario,
                maximum_seconds=10,
            )

    constructed_second = complete_candidate.trace["play"]["plays"][1]

    def later_divergence(*args, **kwargs):
        kwargs["checkpoint_callback"]({
            "schemaVersion": 1,
            "kind": "quality_decision_checkpoint_v1",
            "stage": "play",
            "decision": {
                "playIndex": 1,
                "actor": constructed_second["actor"],
                "action": different_first_action,
                "observerInformationHash": "1" * 64,
            },
        })
        raise ContinuedAfterDivergence

    with monkeypatch.context() as later_patch:
        later_patch.setattr(
            executor_module,
            "resolve_quality_round_continuation_v1",
            later_divergence,
        )
        with pytest.raises(ContinuedAfterDivergence):
            OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
                executor, complete_candidate, scenario=target_scenario,
                maximum_seconds=10,
            )

    def mismatched_quality(*args, **kwargs):
        kwargs["checkpoint_callback"]({
            "schemaVersion": 1,
            "kind": "quality_decision_checkpoint_v1",
            "stage": "play",
            "decision": {"playIndex": 0, "actor": constructed_first["actor"],
                         "action": different_first_action},
        })
        kwargs["checkpoint_callback"]({
            "schemaVersion": 1,
            "kind": "quality_decision_checkpoint_v1",
            "stage": "play",
            "decision": {"playIndex": target, "actor": other_actor},
        })
        pytest.fail("known decision-point miss continued quality generation")

    with monkeypatch.context() as target_patch:
        target_patch.setattr(
            executor_module, "resolve_quality_round_continuation_v1", mismatched_quality,
        )
        early = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
            executor, complete_candidate, scenario=target_scenario, maximum_seconds=10,
        )
    assert early.status == "not_found"
    assert early.reason == "quality_generation_target_miss"
    assert early.partial_receipt["reason"] == "decision_point_target_miss"
    assert early.partial_receipt["targetPlayIndex"] == target
    assert early.partial_receipt["targetActor"] == target_actor
    assert early.partial_receipt["firstConstructedPlayDivergenceIndex"] == 0
    assert early.partial_receipt["generatedActorAtTarget"] == other_actor
    assert early.partial_receipt["qualityCheckpoint"]["decision"] == {
        "playIndex": target, "actor": other_actor,
    }
    assert early.partial_receipt["qualityElapsedSeconds"] >= 0
    assert early.partial_receipt["sha256"] == canonical_sha256({
        key: value for key, value in early.partial_receipt.items() if key != "sha256"
    })

    later_target = _scenario(
        constraints=scenario.constraints.to_dict(), outcome=scenario.outcome.to_dict(),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "position": 3, "remainingCards": 2, "actingSeat": target_actor}},
    )
    def mismatched_trick_leader(*args, **kwargs):
        kwargs["checkpoint_callback"]({
            "kind": "quality_decision_checkpoint_v1", "stage": "play",
            "decision": {"playIndex": 24, "actor": executor.rules.structural_domain.offset(
                Seat.parse(other_actor), -3).name.lower()},
        })
        pytest.fail("known same-trick actor mismatch spent more proof work")
    with monkeypatch.context() as leader_patch:
        leader_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", mismatched_trick_leader)
        predicted = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
            executor, complete_candidate, scenario=later_target, maximum_seconds=10)
    assert predicted.reason == "quality_generation_target_miss"
    assert predicted.partial_receipt["generatedActorAtTarget"] == other_actor

    if source_eligible:
        future_leaf = _leaf("PRED-FUTURE-EXACT-CARD", "play.trick_card_count", {
            "trickIndex": 7, "selector": {"cards": [Card.parse(constructed_first["card"]).id]},
            "minimum": 1, "maximum": 1})
        future_scenario = _scenario(constraints=future_leaf,
            outcome=_leaf("PRED-FUTURE-REPORT", "state.phase", {"state": "playing"}, "report"),
            decision_point=target_scenario.decision_point)
        next_document = executor_module._construction_auction_prefix_v1(complete_candidate).to_dict()
        next_document.pop("sha256")
        next_document["playActions"] = [constructed_first]

        def spent_future_card(*args, **kwargs):
            kwargs["checkpoint_callback"]({"kind": "quality_decision_checkpoint_v1", "stage": "play",
                "nextPrefix": next_document,
                "decision": {"playIndex": 0, "actor": constructed_first["actor"],
                             "action": constructed_first["card"]}})
            raise ContinuedAfterDivergence

        with monkeypatch.context() as future_patch:
            future_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", spent_future_card)
            missed = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
                executor, complete_candidate, scenario=future_scenario, maximum_seconds=10)
        assert missed.status == "not_found" and missed.reason == "quality_generation_target_miss"
        assert missed.partial_receipt["reason"] == "hard_future_card_count_miss"
        assert missed.partial_receipt["futureCardCountPrecheck"]["solverStatus"] == "infeasible"
        assert missed.partial_receipt["futureCardCountPrecheck"]["predicateIds"] == ["PRED-FUTURE-EXACT-CARD"]

        optional_future = _scenario(constraints={"op": "any", "children": [future_leaf,
            _leaf("PRED-FUTURE-ALTERNATIVE", "state.phase", {"state": "playing"})]},
            outcome=future_scenario.outcome.to_dict(), decision_point=target_scenario.decision_point)
        with monkeypatch.context() as optional_patch:
            optional_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", spent_future_card)
            with pytest.raises(ContinuedAfterDivergence):
                OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
                    executor, complete_candidate, scenario=optional_future, maximum_seconds=10)

        next_document["auctionActions"] = [{"actor": "north", "action": "pass"},
                                            {"actor": "west", "action": "sun"}]
        with monkeypatch.context() as changed_auction_patch:
            changed_auction_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", spent_future_card)
            with pytest.raises(ContinuedAfterDivergence):
                OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
                    executor, complete_candidate, scenario=future_scenario, maximum_seconds=10)

    project_scenario = _scenario(
        constraints=_leaf("PRED-PROJECT", "project.exists", {
            "team": "west_east", "kind": "hundred_sequence", "value": 20,
            "state": "awarded",
        }),
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
        }},
    )

    def missed_award(*args, **kwargs):
        kwargs["checkpoint_callback"]({
            "kind": "quality_decision_checkpoint_v1", "stage": "project",
            "publicAwardedProjects": [],
        })
        kwargs["checkpoint_callback"]({
            "kind": "quality_decision_checkpoint_v1", "stage": "play",
            "decision": {"playIndex": 3}, "publicAwardedProjects": [],
        })
        kwargs["checkpoint_callback"]({
            "kind": "quality_decision_checkpoint_v1", "stage": "play",
            "decision": {"playIndex": 7},
            "publicAwardedProjects": [{
                "owner": "north", "team": "north_south",
                "kind": "hundred_sequence", "value": 20,
            }],
        })
        pytest.fail("known project award miss continued into card play")

    with monkeypatch.context() as project_patch:
        project_patch.setattr(
            executor_module, "resolve_quality_round_continuation_v1", missed_award,
        )
        award_miss = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
            executor, complete_candidate, scenario=project_scenario,
            maximum_seconds=10,
        )
    assert award_miss.status == "not_found"
    assert award_miss.partial_receipt is not None, award_miss.reason
    assert award_miss.partial_receipt["reason"] == "hard_project_award_miss"
    assert award_miss.partial_receipt["projectAwardMiss"]["requiredPredicateIds"] == [
        "PRED-PROJECT"
    ]
    assert award_miss.partial_receipt["qualityCheckpoint"]["decision"]["playIndex"] == 7
    assert award_miss.partial_receipt["qualityCheckpoint"]["publicAwardedProjects"] == (
        award_miss.partial_receipt["projectAwardMiss"]["publicAwardedProjects"]
    )

    def direct_selector_miss(*args, **kwargs):
        return {
            "status": "incomplete",
            "finalPrefix": {"playActions": [{"actor": other_actor, "card": "CA"}]},
            "executionReceipt": {"auction": {"elapsedSeconds": 2.0},
                "postAuction": {"projectPlay": {"elapsedSeconds": 3.0}}},
            "partialReceipt": {
                "exhaustionReason": "featured_decision_mismatch",
                "evidence": {"actualActor": other_actor},
            },
        }

    with monkeypatch.context() as mismatch_patch:
        mismatch_patch.setattr(
            executor_module,
            "resolve_quality_round_continuation_v1",
            direct_selector_miss,
        )
        direct = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
            executor,
            complete_candidate,
            scenario=target_scenario,
            maximum_seconds=10,
        )
    assert direct.status == "not_found"
    assert direct.reason == "quality_generation_target_miss"
    assert direct.partial_receipt["generatedActorAtTarget"] == other_actor
    assert direct.partial_receipt["qualityEvidence"]["finalPrefix"]["playActions"][0]["actor"] == other_actor
    assert direct.stage_timings["qualityAuction"] == 2.0
    assert direct.stage_timings["qualityProjectPlay"] == 3.0

    def recorded_target_miss(*args, **kwargs):
        raise executor_module._QualityScenarioTargetMiss("recorded escalation diverged")

    with monkeypatch.context() as recorded_patch:
        recorded_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", recorded_target_miss)
        recorded = executor_module._construction_complete_prefix_v1(complete_candidate)
        audited = OrToolsUnifiedResearchExecutorV1.audit_quality_candidate(
            executor, complete_candidate, recorded_trace=recorded, recorded_events=(),
            decision_point={"kind": "escalation", "selector": {}},
            continue_recorded_prefix=True, maximum_seconds=10)
    assert audited.status == "not_found"
    assert audited.reason == "quality_recorded_audit_target_miss"
    assert audited.partial_receipt["detail"] == "recorded escalation diverged"

    cancelled = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
        executor, complete_candidate,
        maximum_seconds=10, cancel_check=lambda: True,
    )
    assert cancelled.status == "incomplete", (
        cancelled.reason,
        cancelled.partial_receipt.get("qualityStatus"),
        cancelled.partial_receipt.get("qualityFailureStage"),
        cancelled.partial_receipt.get("qualityFailureReason"),
    )
    assert cancelled.reason == "quality_generation_incomplete"
    assert cancelled.partial_receipt["qualityFailureReason"] == "cancelled"


@pytest.mark.parametrize("opening,twice,impossible", [
    (None, False, False), ("DQ", False, False), ("CA", False, True), (None, True, True),
])
def test_future_card_count_relaxation_preserves_remaining_aces(monkeypatch, opening, twice, impossible):
    hands = {"north": ["SK", "H10", "DQ", "C10", "CJ", "CQ", "CK", "CA"],
        "west": ["S7", "S10", "H9", "D10", "DK", "DA", "C8", "C9"],
        "south": ["S9", "SQ", "SA", "H7", "HJ", "HQ", "HK", "C7"],
        "east": ["S8", "SJ", "H8", "HA", "D7", "D8", "D9", "DJ"]}
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "dealer": "east",
        "deckOrder": [11, 21, 27, 0, 22, 23, 2, 5, 7, 1, 16, 17, 28, 31, 25, 26,
                      8, 24, 18, 20, 6, 29, 30, 3, 10, 19, 12, 13, 14, 4, 9, 15],
        "matchScore": [0, 0], "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [], "projectDeclarations": [],
        "playActions": [{"actor": "north", "card": opening}] if opening else []})
    leaves = [_leaf(f"PRED-FUTURE-ACES-{trick}", "play.trick_card_count", {
        "trickIndex": trick, "selector": {"ranks": ["ace"],
            "owner": {"relation": "team", "team": "north_south"}}, "minimum": 2, "maximum": 2})
        for trick in ((6, 7) if twice else (7,))]
    predicates = tuple(_scenario(constraints={"op": "all", "children": leaves},
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report")).predicates())
    predicates = tuple(row for row in predicates if row.kind == "play.trick_card_count")
    result = executor_module._future_card_count_relaxation_v1(hands, prefix, predicates, maximum_seconds=5)
    assert (result is not None) == impossible
    if result is not None:
        assert result["solverStatus"] == "infeasible"
        assert result["sourcePrefixHash"] == prefix.sha256
        assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
    assert executor_module._future_card_count_relaxation_v1(hands, prefix, predicates,
        maximum_seconds=5, cancel_check=lambda: True) is None
    assert executor_module._future_card_count_relaxation_v1(hands, prefix, predicates,
        maximum_seconds=0) is None
    monkeypatch.setattr(executor_module, "_solve_cp_sat_with_cancel_v1", lambda *_args: cp_model.UNKNOWN)
    assert executor_module._future_card_count_relaxation_v1(hands, prefix, predicates, maximum_seconds=5) is None


@pytest.mark.real_clock
def test_quality_generation_keeps_terminal_state_private(
    root, complete_candidate, monkeypatch,
):
    from baloot_platform.research import result_authority
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted", "decisions": {
            kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
                   "horizon": {"unit": "plies", "value": 1},
                   "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in SCENARIO_DECISION_KINDS
        }})
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3", bundle_revision_id="CBR-PRIVATE-HANDOFF",
        bundle_hash="a" * 64, quality_policy=policy,
        quality_execution=ScenarioQualityExecutionBudgetV1.build({
            "schemaVersion": 1, "maximumWallSeconds": 30,
            "maximumEvidenceBytes": 1_000_000, "maximumEscalationDecisions": 20,
            "escalationOpportunityOrder": ["west", "east", "south", "north"]}),
        quality_proof_profiles={"test": {
            "maximumNodes": 100, "maximumDepth": 8, "maximumHiddenWorlds": 1,
            "maximumHorizonActions": 8, "maximumWallSeconds": 5,
            "maximumCpuSeconds": 5, "maximumCacheEntries": 100,
            "maximumCacheBytes": 1_000_000, "cancellationCheckIntervalNodes": 1,
            "maximumWorkers": 1, "maximumLineCountNodes": 100,
            "maximumStateSpace": 100}},
    )
    trace = complete_candidate.trace
    project_actions = [dict(row) for row in trace["projects"].get("declarationActions", ())]
    declared_projects = sorted(
        (row for row in trace["projects"]["projects"] if row["declared"]),
        key=lambda row: int(row["declarationOrder"]),
    )
    final_prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "deckOrder": list(trace["deckOrder"]),
        "dealer": trace["dealer"], "roundNumber": int(trace.get("roundNumber", 1)),
        "matchScore": list(trace.get("matchScore", (0, 0))),
        "auctionActions": [{key: row[key] for key in ("actor", "action", "trumpSuit")
                            if row.get(key) is not None}
                           for row in trace["auction"]["completion"]["actions"]],
        "escalationActions": [{key: row[key] for key in ("actor", "action", "requestedMode")
                               if row.get(key) is not None}
                              for row in trace["escalation"]["observation"]["actions"]],
        **({"projectActions": project_actions} if project_actions else {}),
        "projectDeclarations": [] if project_actions else [{
            key: row[key] for key in (
                "owner", "kind", "value", "highRank", "declarationOrder")
        } for row in declared_projects],
        "projectAssertions": [],
        "playActions": [{"actor": row["actor"], "card": row["card"],
                         "ekka": row.get("ekka", False)} for row in trace["play"]["plays"]],
        "stateAssertions": {},
    }).to_dict()
    canonical_prefix = ExactTracePrefixV1.build(final_prefix)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        canonical_prefix, decision_point={"kind": "play", "selector": {}},
        maximum_seconds=30,
    )
    assert canonical_prefix.sha256 != normalized.prefix_hash
    assert normalized.prefix_hash == executor_module._effective_prefix_hash(canonical_prefix)
    evidence = {"receipt": {
        "status": "complete", "operation": "generate",
        "finalPrefix": canonical_prefix.to_dict(),
        "postAuction": {"projectPlay": {"finalExactStateId": normalized.exact_state_id}},
    }}
    normalized_candidate = executor._candidate_from_terminal_quality_trace(
        {"playingQualityEvidence": evidence}, canonical_prefix, normalized, origin="search",
    )
    assert normalized_candidate.exact_state_id == normalized.exact_state_id
    changed = canonical_prefix.to_dict()
    changed.pop("sha256")
    changed["playActions"][-1]["ekka"] = True
    with pytest.raises(ValueError, match="completed trace: prefix_hash"):
        executor._candidate_from_terminal_quality_trace(
            {"playingQualityEvidence": evidence}, ExactTracePrefixV1.build(changed),
            normalized, origin="search",
        )
    terminal = object()
    round_result = {"status": "complete", "finalPrefix": final_prefix,
                    "executionReceipt": {"elapsedSeconds": 0, "evidenceBytes": 0,
                                         "auction": {"elapsedSeconds": 0},
                                         "postAuction": None}}
    validations = []
    built = []

    def retained_round(*args, **kwargs):
        # An outcome predicate must not freeze the incidental construction bids.
        assert kwargs["search_generated"] is True
        assert kwargs["recorded_trace"] is None and kwargs["recorded_events"] is None
        assert kwargs["continue_recorded_prefix"] is False
        first_play = trace["play"]["plays"][0]
        kwargs["checkpoint_callback"]({
            "kind": "quality_decision_checkpoint_v1", "stage": "play",
            "decision": {"playIndex": 0, "actor": first_play["actor"],
                         "action": "deliberately-different"},
        })
        kwargs["final_normalized_callback"](terminal)
        return round_result

    def build(_self, trace, prefix, normalized, *, origin=None):
        built.append((trace, prefix, normalized, origin))
        return complete_candidate

    monkeypatch.setattr(executor_module, "resolve_quality_round_continuation_v1", retained_round)
    monkeypatch.setattr(executor_module, "validate_trace_quality_evidence_v1",
                        lambda *args, **kwargs: validations.append((args, kwargs)))
    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1,
                        "_candidate_from_terminal_quality_trace", build)

    result = executor.generate_quality_candidate(
        complete_candidate, scenario=_matching_scenario(complete_candidate),
        maximum_seconds=30,
    )

    assert result.status == "feasible" and result.candidate is complete_candidate
    assert result.graph is None and result.proof_outcome is None
    assert result.proof_state_projection_hash is None
    assert not validations and len(built) == 1  # Feasible is not qualified; _result owns admission.
    assert built[0][2] is terminal
    assert "_finalNormalized" not in json.dumps(round_result, sort_keys=True)

    for reason in ("short", "expired"):
        changed_result = copy.deepcopy(round_result)
        if reason == "short":
            changed_result["finalPrefix"]["playActions"].pop()
        finished = []
        clock = executor_module.time.perf_counter
        def completed(*args, **kwargs):
            finished.append(True)
            return changed_result
        def reject_untrusted(*args, **kwargs):
            raise ValueError("untrusted early-return evidence")
        with monkeypatch.context() as boundary:
            boundary.setattr(executor_module, "resolve_quality_round_continuation_v1", completed)
            boundary.setattr(result_authority, "_validate_trace_quality_evidence_v1", reject_untrusted)
            boundary.setattr(executor_module.time, "perf_counter",
                lambda: clock() + (31 if finished and reason == "expired" else 0))
            if reason == "expired":
                # No fresh readback may start after expiry. Keep diagnostic
                # evidence explicitly unverified, never a feasible candidate.
                expired = executor.generate_quality_candidate(complete_candidate, maximum_seconds=30)
                assert expired.status == "incomplete" and expired.candidate is None
                assert expired.partial_receipt["qualityFailureReason"] == "maximum_seconds"
                assert expired.partial_receipt["qualityValidationIncomplete"]["stage"] == "quality_readback"
                assert expired.partial_receipt["qualityEvidence"]["finalPrefix"] == changed_result["finalPrefix"]
            else:
                with pytest.raises(ValueError, match="untrusted early-return evidence"):
                    executor.generate_quality_candidate(complete_candidate, maximum_seconds=30)

    for status, expected in (("quality_failed", "not_found"), ("incomplete", "incomplete")):
        checked = []
        rejected = {**round_result, "status": status,
                    "partialReceipt": {"stage": "action_analysis", "reason": "recorded_action_not_optimal"}}
        with monkeypatch.context() as boundary:
            boundary.setattr(executor_module, "resolve_quality_round_continuation_v1",
                             lambda *args, **kwargs: rejected)
            boundary.setattr(result_authority, "_validate_trace_quality_evidence_v1",
                             lambda *args, **kwargs: checked.append(True))
            boundary.setattr(executor_module.time, "perf_counter", lambda: 31 if checked else 0)
            result = executor.generate_quality_candidate(complete_candidate, maximum_seconds=30)
        assert checked == [True]
        assert result.status == expected
        assert result.candidate is None
        assert result.partial_receipt["qualityStatus"] == status

    monkeypatch.setattr(executor_module, "resolve_quality_round_continuation_v1",
                        lambda *args, **kwargs: round_result)
    with pytest.raises(ValueError, match="did not retain one terminal state"):
        executor.generate_quality_candidate(complete_candidate, maximum_seconds=30)


@pytest.mark.real_clock
@pytest.mark.parametrize("continue_prefix,stop_reason", [(False, None), (True, None),
    (True, "maximum_seconds"), (True, "cancelled")])
def test_recorded_candidate_cancellation_preserves_full_trace(root, continue_prefix, stop_reason, monkeypatch):
    from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
    backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity={
        "bundleRevisionId": "CBR-QUALITY-CANCEL-TEST", "semanticBundleHash": "a" * 64,
        "effectiveRuntimeConfigHash": "b" * 64})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1",
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} for kind in SCENARIO_DECISION_KINDS}})
    raw = _optional_project_prefix_v1()
    raw["projectActions"] = []
    raw["playActions"] = [{"actor": seat, "card": card} for card, seat in
        zip(range(32), ("north", "east", "south", "west") * 8)]
    if continue_prefix:
        raw["playActions"] = []
        raw["projectActions"] = [{"actor": "north", "action": "pass"}]
    recorded = ExactTracePrefixV1.build(raw)
    candidate = SimpleNamespace(trace=recorded.to_dict(), origin="exact_trace_prefix")
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-QUALITY-CANCEL-TEST", bundle_hash="a" * 64,
        quality_policy=policy, quality_execution=backend.authority.quality_execution,
        quality_proof_profiles=profile["scenarioResearchV4"]["proofBudgets"]["profiles"])
    result = executor.audit_quality_candidate(candidate, recorded_trace=recorded, recorded_events=[],
        decision_point={"kind": "escalation", "selector": {}}, maximum_seconds=30,
        cancel_check=lambda: True, continue_recorded_prefix=continue_prefix)
    assert result.status == "incomplete"
    assert result.candidate is None
    assert "qualityEvidence" not in result.partial_receipt  # Cancelled before any solver audit.
    assert result.partial_receipt["sourcePrefix"]["playActions"] == []
    assert result.partial_receipt["recordedEscalationOpportunities"] == []
    assert result.reason == "quality_recorded_audit_incomplete"
    assert result.partial_receipt["qualityFailureReason"] == "cancelled"
    retained = result.partial_receipt["recordedTrace"]
    assert retained == {"completeExactTracePrefix": recorded.to_dict()}
    assert result.partial_receipt["sha256"] == canonical_sha256({
        key: value for key, value in result.partial_receipt.items() if key != "sha256"})
    assert candidate.trace == recorded.to_dict()
    if continue_prefix:
        calls = []
        def audit(original, **kwargs):
            calls.append(kwargs)
            assert kwargs["recorded_trace"] == recorded
            assert kwargs["recorded_events"] == ()
            assert kwargs["continue_recorded_prefix"] is True
            return result
        def forbidden(*args, **kwargs):
            raise AssertionError("exact-input campaign must never regenerate the submitted prefix")
        monkeypatch.setattr(executor, "audit_quality_candidate", audit)
        monkeypatch.setattr(executor, "generate_quality_candidate", forbidden)
        scenario = _scenario(
            constraints=_leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            outcome=_leaf("PRED-OUTCOME", "outcome.team_round", {"team": "north_south", "status": "won"}, "report"),
            origin={"kind": "exact_trace_prefix", "exactTracePrefix": recorded.to_dict(),
                    "recordedEscalationOpportunities": []},
            decision_point={"kind": "play", "selector": {}})
        ready = []
        original = executor._candidate_from_normalized_exact
        def candidate_checkpoint(*args, **kwargs):
            value = original(*args, **kwargs)
            ready.append(value)
            return value
        monkeypatch.setattr(executor, "_candidate_from_normalized_exact", candidate_checkpoint)
        if stop_reason is not None:
            clock = executor_module.time.perf_counter
            monkeypatch.setattr(executor_module.time, "perf_counter",
                lambda: clock() + (31 if ready and stop_reason == "maximum_seconds" else 0))
        progress = []
        campaign = executor.execute(scenario, _bounded_proof(), maximum_seconds=30,
            maximum_proof_nodes=1000, maximum_proof_depth=8,
            progress_callback=progress.append,
            cancel_check=lambda: bool(ready) and stop_reason == "cancelled")
        checkpoint = progress[-1]
        assert checkpoint["currentStage"] == "playingQuality"
        assert checkpoint["completedCandidates"] == 1
        assert checkpoint["qualifiedCandidates"] == 0
        assert checkpoint["proofCompletionStatus"] == "incomplete"
        assert checkpoint["candidateHash"] == ready[0].sha256
        if stop_reason is not None:
            assert not calls
            assert campaign.status == "incomplete"
            receipt = campaign.execution_receipt["incompleteCompilation"]
            assert receipt["exhaustionReason"] == stop_reason
            assert receipt["candidateHash"] == ready[0].sha256
            assert receipt["stage"] == "playingQuality"
            assert receipt["sha256"] == canonical_sha256({
                key: value for key, value in receipt.items() if key != "sha256"})
            return
        assert calls
        assert campaign.status == "incomplete"
        assert campaign.execution_receipt["incompleteCompilation"]["recordedTrace"] == retained


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


@pytest.mark.real_clock
@pytest.mark.parametrize("mode", ["exact", "bounded"])
def test_complete_state_quality_generates_and_audits_legal_play_suffix(root, rules, mode):
    prefix, actor_request = _quality_continuation_input(root, mode)
    request = ScenarioProofRequestV1.build({
        **{key: value for key, value in actor_request.to_dict().items() if key != "sha256"},
        "visibility": "complete_state",
    })
    limits = dict(maximum_nodes=100_000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=60, maximum_decision_seconds=20, maximum_evidence_bytes=30_000_000)
    result = resolve_quality_play_continuation_v1(rules, prefix, request, **limits)
    assert result["status"] == "complete", result.get("partialReceipt")
    assert len(result["finalPrefix"]["playActions"]) == 32
    assert len(result["decisions"]) == 32 - len(prefix.play_actions)
    for row in result["decisions"]:
        assert row["selectionReason"] == "native_complete_state_best_value"
        assert row["actualActionOptimal"] is True and row["regret"] == 0
        artifact = result["artifacts"][row["proofArtifactHash"]]
        assert artifact["request"]["visibility"] == "complete_state"
        assert artifact["proofModel"]["actionQuality"]["scope"] == "compiled_complete_state"
        assert artifact["proof"]["completionStatus"] == ("exact_complete" if mode == "exact" else "bounded_complete")
        assert "hiddenWorlds" not in artifact["allocationModel"]
    audited = resolve_quality_play_continuation_v1(rules, prefix, request,
        recorded_trace=ExactTracePrefixV1.build(result["finalPrefix"]), **limits)
    assert audited["status"] == "complete", audited.get("partialReceipt")
    assert audited["finalPrefix"] == result["finalPrefix"]
    assert all(row["selectionReason"] == "recorded_action" for row in audited["decisions"])
    stopped = resolve_quality_play_continuation_v1(rules, prefix, request,
        cancel_check=lambda: True, **limits)
    assert stopped["status"] == "incomplete" and stopped["decisions"] == []


def test_raw_editorial_companion_is_native_owned_and_does_not_replace_quality(root, monkeypatch):
    from baloot_platform.research.result_authority import (
        _validate_native_decision_quality_evidence_v1, native_raw_point_quality_request_v1,
    )
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix, template = _quality_continuation_input(root, "exact")
    prefix_raw = prefix.to_dict()
    prefix_raw.pop("sha256")
    prefix_raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(prefix_raw)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    profile = "late_game_exact_v1"
    budget = ScenarioProofBudgetV1.build(profile, settings["proofBudgets"]["profiles"][profile])
    raw = template.to_dict()
    raw.pop("sha256")
    raw.update(visibility="complete_state", budgetProfile=profile)
    request = ScenarioProofRequestV1.build(raw)
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": budget.maximum_wall_seconds, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20, "exactPlayFromTrick": 1,
        "videoInterestingness": {"schemaVersion": 2, "maximumFinalMargin": 40,
            "minimumPointSwing": 15, "maximumFeaturedTrick": 6, "minimumMeaningfulDecisions": 1}})
    kwargs = dict(maximum_nodes=budget.maximum_nodes, maximum_depth=budget.maximum_depth,
        maximum_worlds=1, maximum_seconds=budget.maximum_wall_seconds,
        maximum_decision_seconds=budget.maximum_wall_seconds, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles=settings["proofBudgets"]["profiles"], execution_budget=execution,
        featured_play_index=len(prefix.play_actions), featured_decision_point={"kind": "play", "selector": {}})
    value = executor_module._resolve_quality_continuation_v1(rules, prefix, request, **kwargs)
    assert value["status"] == "complete", value["partialReceipt"]
    assert len(value["rawPointEvidence"]) == 1
    digest, companion = next(iter(value["rawPointEvidence"].items()))
    source = value["artifacts"][digest]
    derived = native_raw_point_quality_request_v1(ScenarioProofRequestV1.build(source["request"]))
    assert derived.sha256 == canonical_sha256({key: item for key, item in derived.to_dict().items() if key != "sha256"})
    assert companion["request"] == derived.to_dict()
    assert companion["proof"]["completionStatus"] == "exact_complete"
    assert source["request"]["utility"] == "awarded_point_margin"
    assert value["executionReceipt"]["rawPointProof"]["evidenceBytes"] > 0
    read = dict(source_prefix=prefix, rules_hash=rules.sha256, request=request, rules=rules,
        execution_budget=execution, maximum_seconds=budget.maximum_wall_seconds)
    _validate_native_decision_quality_evidence_v1(value, **read)
    paired_raw = execution.to_dict()
    paired_raw.pop("sha256")
    paired_execution = ScenarioQualityExecutionBudgetV1.build({**paired_raw, "completeStatePlayPolicyPair": True})
    with monkeypatch.context() as patch:
        def forbid_pair(*args, **kwargs):
            pytest.fail("recorded descendants must retain independent exact roots after the focal")
        patch.setattr(executor_module.OrToolsActorInformationSetCompilerV1, "_synthesize_complete_state_policy_pair", forbid_pair)
        audited = executor_module._resolve_quality_continuation_v1(rules, prefix, request,
            **{**kwargs, "execution_budget": paired_execution,
               "recorded_trace": ExactTracePrefixV1.build(value["finalPrefix"]), "stop_on_recorded_quality_failure": True})
    assert audited["status"] == "complete", audited.get("partialReceipt")
    assert audited["finalPrefix"]["playActions"] == value["finalPrefix"]["playActions"]
    assert all(artifact["proof"]["completionStatus"] == "exact_complete" for artifact in audited["artifacts"].values())
    _validate_native_decision_quality_evidence_v1(audited, **{**read, "execution_budget": paired_execution})
    for field in ("sourcePrefixHash", "exactStateId"):
        changed = copy.deepcopy(value)
        changed["rawPointEvidence"][digest]["allocationModel"][field] = "substituted"
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError):
            _validate_native_decision_quality_evidence_v1(changed, **read, verified_artifacts=set(value["artifacts"]))
    changed = copy.deepcopy(value)
    changed["rawPointEvidence"]["orphan"] = changed["rawPointEvidence"].pop(digest)
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="source reference"):
        _validate_native_decision_quality_evidence_v1(changed, **read, verified_artifacts=set(value["artifacts"]))
    changed = copy.deepcopy(value)
    changed["limits"]["maximumDepth"] = 1
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="depth budget"):
        _validate_native_decision_quality_evidence_v1(changed, **read, verified_artifacts=set(value["artifacts"]))
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, _integral_success_threshold,
    )
    from baloot_platform.research.result_authority import build_native_complete_state_play_quality_artifact_v1
    graph = copy.deepcopy(companion["graph"])
    def rehash(document):
        document["sha256"] = canonical_sha256({key: item for key, item in document.items() if key != "sha256"})
    for proof in graph["roots"]["roots"].values():
        proof["value"] += 1
        rehash(proof)
    rehash(graph["roots"])
    graph["certificate"] = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._certify_observed_root_actions(
        graph["roots"], success_threshold=_integral_success_threshold(derived.success_threshold),
        claim=derived.claim, selected_action=derived.selected_action,
        maximum_seconds=budget.maximum_wall_seconds, action_quality=True, quality_scope="compiled_complete_state")
    graph["status"] = graph["certificate"]["status"]
    rehash(graph)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=budget.maximum_wall_seconds)
    changed = copy.deepcopy(value)
    changed["rawPointEvidence"][digest] = build_native_complete_state_play_quality_artifact_v1(
        graph, rules=rules, source_prefix=prefix, normalized=normalized, request=derived, budget=budget)
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="differs from fresh native proof"):
        _validate_native_decision_quality_evidence_v1(changed, **read, verified_artifacts=set(value["artifacts"]))
    original = executor_module._complete_state_quality_artifact_v1
    def stop_raw(*args, **limits):
        if args[4].utility == "_internal_raw_point_margin_v1":
            from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
            raise InformationSetCompilationStopped("maximum_seconds", stage="raw_point_editorial_proof")
        return original(*args, **limits)
    monkeypatch.setattr(executor_module, "_complete_state_quality_artifact_v1", stop_raw)
    incomplete_raw = executor_module._resolve_quality_continuation_v1(rules, prefix, request, **kwargs)
    assert incomplete_raw["status"] == "complete" and "rawPointEvidence" not in incomplete_raw
    assert incomplete_raw["executionReceipt"]["rawPointProof"]["status"] == "incomplete"
    assert "partialReceipt" not in incomplete_raw["executionReceipt"]["rawPointProof"]
    def cancel_raw(*args, **limits):
        if args[4].utility == "_internal_raw_point_margin_v1":
            from baloot_platform.research.ortools_exact_state import ExactNormalizationStopped
            raise ExactNormalizationStopped("cancelled")
        return original(*args, **limits)
    monkeypatch.setattr(executor_module, "_complete_state_quality_artifact_v1", cancel_raw)
    with pytest.raises(executor_module.ExactNormalizationStopped):
        executor_module._resolve_quality_continuation_v1(rules, prefix, request, **kwargs)


@pytest.mark.parametrize("play_index,expected_cap", [(23, 2048), (24, 256)])
def test_quality_world_cap_dispatch_uses_actual_play_index(root, monkeypatch, play_index, expected_cap):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix, request = _quality_continuation_input(root, "exact")
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["playActions"] = raw["playActions"][:play_index]
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(raw)
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 30, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20, "maximumHiddenWorldsPerDecision": 2048,
        "playIndexWorldCaps": {"schemaVersion": 1, "bands": [
            {"firstPlayIndex": 16, "lastPlayIndex": 23, "maximumHiddenWorlds": 2048},
            {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": 256}]}})
    class DispatchCaptured(BaseException):
        pass
    def capture(*args, **kwargs):
        assert len(args[2].play_actions) == play_index
        assert kwargs["maximum_worlds"] == expected_cap
        raise DispatchCaptured
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", capture)
    with pytest.raises(DispatchCaptured):
        resolve_quality_play_continuation_v1(rules, prefix, request,
            maximum_nodes=1000, maximum_depth=12, maximum_worlds=4096,
            maximum_seconds=30, maximum_decision_seconds=30,
            maximum_evidence_bytes=30_000_000, execution_budget=execution)


@pytest.mark.parametrize("execution_world_cap", [8, 256, None, "bands"])
def test_actor_visible_raw_companion_uses_exhaustive_worlds_and_fresh_readback(root, record_property, execution_world_cap, monkeypatch):
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix, template = _quality_continuation_input(root, "exact")
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(raw)
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    profile = "actor_information_set_exact_v1"
    settings["proofBudgets"]["profiles"][profile]["maximumHorizonActions"] = 4
    budget = ScenarioProofBudgetV1.build(profile, settings["proofBudgets"]["profiles"][profile])
    raw = template.to_dict()
    raw.pop("sha256")
    raw.update(budgetProfile=profile)
    request = ScenarioProofRequestV1.build(raw)
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": budget.maximum_wall_seconds, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20, "exactPlayFromTrick": 7,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        **({"maximumHiddenWorldsPerDecision": 2048, "playIndexWorldCaps": {"schemaVersion": 1, "bands": [
            {"firstPlayIndex": 16, "lastPlayIndex": 23, "maximumHiddenWorlds": 2048},
            {"firstPlayIndex": 24, "lastPlayIndex": 31, "maximumHiddenWorlds": 256}]}}
            if execution_world_cap == "bands" else
            {"maximumHiddenWorldsPerDecision": execution_world_cap} if execution_world_cap is not None else {}),
        "exactTailProof": True,
        "videoInterestingness": {"schemaVersion": 2, "maximumFinalMargin": 40,
            "minimumPointSwing": 15, "maximumFeaturedTrick": 6, "minimumMeaningfulDecisions": 1}})
    observed_limits = []
    compile_observed = executor_module.compile_observed_decision_v1
    def capture_world_limit(*args, **kwargs):
        observed_limits.append((len(args[2].play_actions), args[4].utility, kwargs["maximum_worlds"]))
        return compile_observed(*args, **kwargs)
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", capture_world_limit)
    value = resolve_quality_play_continuation_v1(rules, prefix, request,
        maximum_nodes=budget.maximum_nodes, maximum_depth=12, maximum_worlds=2048,
        maximum_seconds=budget.maximum_wall_seconds, maximum_decision_seconds=budget.maximum_wall_seconds,
        maximum_evidence_bytes=30_000_000, proof_budget_profiles=settings["proofBudgets"]["profiles"],
        execution_budget=execution, featured_play_index=len(prefix.play_actions),
        featured_decision_point={"kind": "play", "selector": {}})
    assert value["status"] == "complete", json.dumps(value.get("partialReceipt"), indent=2)
    assert value["limits"]["maximumWorldsPerDecision"] == 2048
    assert {utility for _, utility, _ in observed_limits} == {
        "awarded_point_margin", "_internal_raw_point_margin_v1"}
    effective_cap = 256 if execution_world_cap == "bands" else execution_world_cap or 2048
    assert {limit for _, _, limit in observed_limits} == {effective_cap}
    digest, companion = next(iter(value["rawPointEvidence"].items()))
    assert value["artifacts"][digest]["request"]["utility"] == "awarded_point_margin"
    assert companion["request"]["visibility"] == "actor_information_set"
    assert companion["request"]["utility"] == "_internal_raw_point_margin_v1"
    assert companion["proofModel"]["kind"] == "actor_information_set_cegis_model_v1"
    assert companion["proof"]["hiddenWorlds"]["complete"] is True
    assert 1 < companion["proof"]["hiddenWorlds"]["count"] <= effective_cap
    assert companion["proof"]["hiddenWorlds"]["coverage"]["method"] == "exhaustive_cp_sat_hidden_allocations_v1"
    record_property("rawHiddenWorlds", companion["proof"]["hiddenWorlds"]["count"])
    record_property("rawPointProofSeconds", value["executionReceipt"]["rawPointProof"]["elapsedSeconds"])
    read = dict(source_prefix=prefix, rules_hash=rules.sha256, request=request, rules=rules,
        execution_budget=execution, maximum_seconds=budget.maximum_wall_seconds)
    from baloot_platform.research import result_authority
    readback_limits = []
    reverify = result_authority.reverify_native_exact_observed_play_artifact_v1
    def capture_readback(*args, **kwargs):
        readback_limits.append((kwargs["request"].utility, kwargs["maximum_worlds"]))
        return reverify(*args, **kwargs)
    monkeypatch.setattr(result_authority, "reverify_native_exact_observed_play_artifact_v1", capture_readback)
    validate_native_decision_quality_evidence_v1(value, **read)
    assert {limit for _, limit in readback_limits} == {effective_cap}
    assert {utility for utility, _ in readback_limits} == {
        "awarded_point_margin", "_internal_raw_point_margin_v1"}
    for section in ("sourcePrefix", "proof"):
        changed = copy.deepcopy(value)
        if section == "sourcePrefix":
            changed["rawPointEvidence"][digest][section]["sha256"] = "0" * 64
        else:
            changed["rawPointEvidence"][digest][section]["rootActions"][0]["value"] += 1
        changed["sha256"] = canonical_sha256({key: item for key, item in changed.items() if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError):
            validate_native_decision_quality_evidence_v1(changed, **read)


@pytest.mark.parametrize("tie_rule", [None, "draw"])
def test_content_interest_rejects_missing_tie_semantics_before_normalization(root, rules, monkeypatch, tie_rule):
    prefix, request = _quality_continuation_input(root, "exact")
    data = copy.deepcopy(rules.data)
    if tie_rule is None:
        data["scoring"]["contract"].pop("buyer_tie_outcome", None)
    else:
        data["scoring"]["contract"]["buyer_tie_outcome"] = tie_rule
    altered = copy.deepcopy(rules)
    altered.data = data
    def no_normalization(*args, **kwargs):
        pytest.fail("invalid interest configuration reached normalization")
    monkeypatch.setattr(executor_module, "OrToolsExactStateNormalizerV1", no_normalization)
    with pytest.raises(ValueError, match="explicit buyer tie rule"):
        resolve_quality_play_continuation_v1(altered, prefix, request,
            execution_budget=SimpleNamespace(video_interestingness={"schemaVersion": 4}),
            maximum_nodes=10, maximum_depth=8, maximum_worlds=1, maximum_seconds=1,
            maximum_decision_seconds=1, maximum_evidence_bytes=1000)


@pytest.mark.parametrize("tie_rule", ["buyer_makes", "buyer_loses"])
def test_content_interest_preflight_accepts_both_explicit_tie_rules(root, rules, tie_rule):
    _, request = _quality_continuation_input(root, "exact")
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = tie_rule
    executor_module.validate_content_interest_rules_v1(RuleBook(data), request,
        SimpleNamespace(video_interestingness={"schemaVersion": 4}))


@pytest.mark.parametrize("version,mode,visibility", [
    (None, "exact", "actor_information_set"), (3, "exact", "actor_information_set"),
    (4, "bounded", "actor_information_set"), (4, "exact", "complete_state"),
])
def test_content_interest_preflight_preserves_ungated_and_legacy_execution(root, rules, version, mode, visibility):
    _, request = _quality_continuation_input(root, mode)
    request = replace(request, visibility=visibility)
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"].pop("buyer_tie_outcome", None)
    executor_module.validate_content_interest_rules_v1(RuleBook(data), request,
        SimpleNamespace(video_interestingness={"schemaVersion": version}) if version else None)


@pytest.mark.parametrize("exact_tail", [False, True])
def test_content_interest_preflight_checks_budgeted_policys_eventual_exact_tail(root, rules, monkeypatch, exact_tail):
    prefix, request = _quality_continuation_input(root, "bounded")
    analysis = {key: value for key, value in request.to_dict().items()
        if key in {"utility", "budgetProfile", "horizon", "boundaryEvaluator"}}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: copy.deepcopy(analysis) for kind in SCENARIO_DECISION_KINDS}})
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 180, "maximumEvidenceBytes": 20_000_000, "maximumEscalationDecisions": 20,
        "exactPlayFromTrick": 7, "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        "exactTailProof": exact_tail, "videoInterestingness": {
            "schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
            "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1,
            "minimumConsistentWorlds": 2, "minimumBaselineRegret": 20, "minimumNextBestRegret": 20,
            "baselinePolicyId": "observer_safe_trick_control_v1", "allowTieRuleOnlyLessons": False}})
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"].pop("buyer_tie_outcome", None)
    legacy_rules = RuleBook(data)
    if not exact_tail:
        executor_module.validate_content_interest_rules_v1(legacy_rules, policy, execution)
        return
    def no_normalization(*args, **kwargs):
        pytest.fail("invalid exact-tail configuration reached normalization")
    monkeypatch.setattr(executor_module, "OrToolsExactStateNormalizerV1", no_normalization)
    with pytest.raises(ValueError, match="explicit buyer tie rule"):
        resolve_quality_play_continuation_v1(legacy_rules, prefix, policy,
            execution_budget=execution, maximum_nodes=10, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=1, maximum_decision_seconds=1, maximum_evidence_bytes=1000)


def _quality_continuation_input(root: Path, mode: str):
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    value = copy.deepcopy(fixture["exactTracePrefix"])
    value.pop("sha256", None)
    value["stateAssertions"] = {}
    value["playActions"].extend(
        {"card": code, "ekka": False}
        for code in fixture["expected"]["principalVariation"][:4]
    )
    request = {
        "schemaVersion": 1, "mode": mode, "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": "north_south", "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "test",
    }
    if mode == "bounded":
        request.update(horizon={"unit": "tricks", "value": 1}, boundaryEvaluator="current_raw_point_margin_v1")
    return ExactTracePrefixV1.build(value), ScenarioProofRequestV1.build(request)


@pytest.mark.real_clock
@pytest.mark.parametrize("action,interrupt_readback", [("SA", False), ("S7", False), ("SA", True)])
def test_late_recorded_precheck_refutes_only_and_preserves_full_audit(
    root, monkeypatch, action, interrupt_readback, record_property,
):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix, _ = _quality_continuation_input(root, "exact")
    raw = prefix.to_dict()
    raw.pop("sha256")
    # Make the last lead a diamond: saving the Ace now feeds our partner's
    # last-trick winner instead of losing it in the opponents' club trick.
    exchange = {Card.parse("C9").id: Card.parse("D9").id,
                Card.parse("D9").id: Card.parse("C9").id}
    raw["deckOrder"] = [exchange.get(card, card) for card in raw["deckOrder"]]
    for row in raw["playActions"]:
        card = Card.parse(row["card"])
        row["card"] = Card(exchange.get(card.id, card.id)).code
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    project_raw = {**raw, "playActions": [], "projectActions": []}
    project_state = normalizer.normalize_play_or_project(
        ExactTracePrefixV1.build(project_raw), maximum_seconds=30)
    declaration = next(row for row in project_state.legal_actions if row["action"] == "declare")
    raw["projectActions"][0] = {key: declaration[key] for key in ("actor", "action", "projectId")}
    prefix = ExactTracePrefixV1.build(raw)
    full = normalizer.append_normalized_action(prefix,
        normalizer.normalize_play_or_project(prefix, maximum_seconds=30), action)
    while len(full.play_actions) < 32:
        state = normalizer.normalize_play_or_project(full, maximum_seconds=30)
        assert len(state.legal_actions) == 1
        full = normalizer.append_normalized_action(full, state, state.legal_actions[0]["code"])
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "exact", "visibility": "complete_state", "decisions": {
            kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1"}
            for kind in SCENARIO_DECISION_KINDS}})
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-LATE-PRECHECK-TEST", bundle_hash="a" * 64, quality_policy=policy,
        quality_execution=ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"]),
        quality_proof_profiles=settings["proofBudgets"]["profiles"],
        quality_search_profiles=settings["searchBudgets"]["profiles"])
    candidate = executor._candidate_from_normalized_exact(
        normalizer.normalize_play_or_project(full, maximum_seconds=30), full)
    checkpoints, opening_calls = [], []

    class OpeningStillRequired(Exception):
        pass

    def opening(*args, **kwargs):
        opening_calls.append(kwargs)
        raise OpeningStillRequired

    monkeypatch.setattr(executor_module, "resolve_quality_round_continuation_v1", opening)
    if interrupt_readback:
        def exhausted(*args, **kwargs):
            raise executor_module.InformationSetCompilationStopped("maximum_seconds", stage="readback")
        monkeypatch.setattr(executor_module, "validate_native_decision_quality_evidence_v1", exhausted)
    started = time.perf_counter()
    kwargs = dict(recorded_trace=full, recorded_events=(), maximum_seconds=30,
        decision_point={"kind": "play", "selector": {"trickIndex": 6, "position": 3}},
        stop_on_recorded_quality_failure=True, checkpoint_callback=checkpoints.append)
    if action == "S7" or interrupt_readback:
        with pytest.raises(OpeningStillRequired):
            executor.audit_quality_candidate(candidate, **kwargs)
        assert 0 < opening_calls[0]["runtime_maximum_seconds"] < 30
        assert checkpoints == []
    else:
        result = executor.audit_quality_candidate(candidate, **kwargs)
        assert result.status == "not_found", result.partial_receipt
        assert result.reason == "quality_recorded_audit_target_miss"
        record_property("nativePrecheckReceipt", json.dumps(result.partial_receipt, sort_keys=True))
        precheck = result.partial_receipt["lateQualityPrecheck"]
        assert precheck["scope"] == "suffix_only"
        assert precheck["recordedPrefixHash"] == full.sha256
        assert precheck["receipt"]["status"] == "quality_failed"
        decision = precheck["receipt"]["decisions"][0]
        assert decision["playIndex"] == 27
        assert decision["action"] == "SA" and decision["actualActionOptimal"] is False
        assert result.partial_receipt["recordedTrace"]["completeExactTracePrefix"] == full.to_dict()
        assert len(checkpoints) == 1 and checkpoints[0]["decision"] == decision
        assert opening_calls == []
        assert result.stage_timings["qualityProjectPlay"] > 0
        checkpoint_count = len(checkpoints)
        cancelled = executor.audit_quality_candidate(candidate, **{**kwargs, "cancel_check": lambda: True})
        assert cancelled.status == "incomplete"
        assert cancelled.partial_receipt["qualityFailureReason"] == "cancelled"
        assert len(checkpoints) == checkpoint_count and opening_calls == []
        invalid_raw = full.to_dict()
        invalid_raw.pop("sha256")
        invalid_raw["stateAssertions"] = {"phase": "playing"}
        with pytest.raises(ValueError, match="assertion"):
            executor.audit_quality_candidate(candidate,
                **{**kwargs, "recorded_trace": ExactTracePrefixV1.build(invalid_raw)})
        assert len(checkpoints) == checkpoint_count and opening_calls == []
    record_property("precheckSeconds", time.perf_counter() - started)


@pytest.mark.parametrize("mutate", [False, True])
def test_complete_state_play_readback_reproves_rehashed_root_values(root, mutate):
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, _integral_success_threshold,
    )
    from baloot_platform.research.result_authority import (
        build_native_complete_state_play_quality_artifact_v1, validate_native_decision_quality_evidence_v1,
    )
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix, template = _quality_continuation_input(root, "exact")
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(raw)
    profile = "late_game_exact_v1"
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    budget = ScenarioProofBudgetV1.build(profile, settings["proofBudgets"]["profiles"][profile])
    request_raw = template.to_dict()
    request_raw.pop("sha256")
    request_raw.pop("strategyClass")
    request_raw.update(visibility="complete_state", budgetProfile=profile)
    request = ScenarioProofRequestV1.build(request_raw)
    value = resolve_quality_play_continuation_v1(rules, prefix, request,
        maximum_nodes=budget.maximum_nodes, maximum_depth=min(budget.maximum_depth, budget.maximum_horizon_actions),
        maximum_worlds=1, maximum_seconds=budget.maximum_wall_seconds,
        maximum_decision_seconds=budget.maximum_wall_seconds, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"])
    assert value["status"] == "complete", value.get("partialReceipt")
    if mutate:
        row = next(row for row in value["decisions"] if value["artifacts"][row["proofArtifactHash"]][
            "proofModel"].get("kind") == "native_complete_state_root_outcome_v1")
        old_hash = row["proofArtifactHash"]
        original = value["artifacts"][old_hash]
        graph = copy.deepcopy(original["graph"])
        def rehash(document):
            document["sha256"] = canonical_sha256({k: v for k, v in document.items() if k != "sha256"})
        for proof in graph["roots"]["roots"].values():
            proof["value"] += 1
            rehash(proof)
        rehash(graph["roots"])
        owned_request = ScenarioProofRequestV1.build(original["request"])
        graph["certificate"] = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._certify_observed_root_actions(
            graph["roots"], success_threshold=_integral_success_threshold(owned_request.success_threshold),
            claim=owned_request.claim, selected_action=owned_request.selected_action,
            maximum_seconds=budget.maximum_wall_seconds, action_quality=True, quality_scope="compiled_complete_state")
        graph["status"] = graph["certificate"]["status"]
        rehash(graph)
        owned_prefix = ExactTracePrefixV1.build(original["allocationModel"]["sourcePrefix"])
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            owned_prefix, maximum_seconds=budget.maximum_wall_seconds)
        changed = build_native_complete_state_play_quality_artifact_v1(graph, rules=rules, source_prefix=owned_prefix,
            normalized=normalized, request=owned_request, budget=budget)
        digest = canonical_sha256(changed)
        value["artifacts"].pop(old_hash)
        value["artifacts"][digest] = changed
        row.update(proofArtifactHash=digest, actionQualityHash=changed["proofModel"]["actionQuality"]["sha256"])
        rehash(row)
        value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="roots differ from fresh native proof"):
            validate_native_decision_quality_evidence_v1(value, source_prefix=prefix, rules_hash=rules.sha256,
                request=request, rules=rules, search_budget_profiles=settings["searchBudgets"]["profiles"])
    else:
        validate_native_decision_quality_evidence_v1(value, source_prefix=prefix, rules_hash=rules.sha256,
            request=request, rules=rules, search_budget_profiles=settings["searchBudgets"]["profiles"])


@pytest.mark.parametrize("dynamic,selector_kind,offset", [
    (False, "all", 0), (True, "next_actor", 1),
    (True, "trick", 0), (True, "all", 0), (True, "past_trick", None),
])
def test_featured_play_uses_origin_owned_match_mode(
    root, monkeypatch, dynamic, selector_kind, offset,
):
    fixture = json.loads((root / "tests/fixtures/research-native-observer-budgeted-round-v3.json").read_text(encoding="utf-8"))
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))["scenarioResearchV4"]
    execution = ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"])
    policy = ScenarioPlayingQualityPolicyV1.build(fixture["playingQuality"])
    profile = settings["proofBudgets"]["profiles"][policy.decisions["play"]["budgetProfile"]]
    recorded = ExactTracePrefixV1.build(fixture["exactTracePrefix"])
    prefix = ExactTracePrefixV1.build({**fixture["exactTracePrefix"], "playActions": list(recorded.play_actions[:24])})
    actor = Seat.parse(recorded.play_actions[24]["actor"])
    selector = (
        {"actingSeat": actor.next().name.lower()} if selector_kind == "next_actor" else
        {"trickIndex": len(prefix.play_actions) // 4} if selector_kind == "trick" else
        {"trickIndex": len(prefix.play_actions) // 4 - 1} if selector_kind == "past_trick" else {}
    )
    selected = []
    original = executor_module._select_quality_play_action_v1

    def capture(*args, **kwargs):
        selected.append((args[2].next_actor, kwargs["featured_now"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "_select_quality_play_action_v1", capture)
    result = resolve_quality_play_continuation_v1(
        rules,
        prefix,
        policy,
        recorded_trace=recorded,
        continue_recorded_prefix=True,
        stop_on_recorded_quality_failure=True,
        execution_budget=execution,
        featured_play_index=len(prefix.play_actions),
        featured_decision_point={"kind": "play", "selector": selector},
        featured_selector_dynamic=dynamic,
        maximum_nodes=profile["maximumNodes"],
        maximum_depth=profile["maximumDepth"],
        maximum_worlds=min(profile["maximumHiddenWorlds"], execution.maximum_hidden_worlds_per_decision),
        maximum_seconds=60,
        maximum_decision_seconds=profile["maximumWallSeconds"],
        maximum_evidence_bytes=execution.maximum_evidence_bytes,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"],
    )

    reason = (result.get("partialReceipt") or {}).get("reason")
    if offset is None:
        assert result["status"] == "model_error" and reason == "featured_decision_not_covered"
    else:
        assert result["status"] == "complete", result.get("partialReceipt")
        assert len(result["decisions"]) == 8
        assert reason != "featured_decision_not_covered"
    assert [index for index, (_, featured) in enumerate(selected) if featured] == ([] if offset is None else [offset])


@pytest.mark.parametrize("maximum_nodes", [5, 6])
def test_observed_worlds_share_remaining_node_budget(root, rules, monkeypatch, maximum_nodes):
    source, request = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[], escalationActions=[], projectActions=[],
               projectDeclarations=[], projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix)
    value = request.to_dict()
    value.pop("sha256")
    value["horizon"] = {"unit": "plies", "value": 1}
    request = ScenarioProofRequestV1.build(value)
    actor = Seat.parse(normalized.next_actor)
    graph = ProofGraphV1.build(scope="bounded", root_node_id="root", nodes=[
        ProofGraphNodeV1.decision("root", actor.team.name.lower(), [
            ProofGraphEdgeV1.build(0, "pass", "a"), ProofGraphEdgeV1.build(1, "sun", "b"),
        ], actor_seat=actor.name.lower()),
        ProofGraphNodeV1.leaf("a", 0, kind="boundary"), ProofGraphNodeV1.leaf("b", 0, kind="boundary"),
    ], boundary_evaluator="current_raw_point_margin_v1")
    limits = []
    def compile_world(*args, **kwargs):
        limits.append(kwargs["maximum_nodes"])
        return SimpleNamespace(graph=graph, decision_observation_ids={"root": "observation"})
    def compile_worlds(self, exact, prefix, decision, request, *, graph_compiler, **kwargs):
        for _ in range(3):
            graph_compiler(exact, prefix, request)
        pytest.fail("third world started after the shared node budget was exhausted")
    monkeypatch.setattr(executor_module, "compile_nonplay_complete_graph_v1", compile_world)
    monkeypatch.setattr(executor_module.OrToolsActorInformationSetCompilerV1, "compile_nonplay", compile_worlds)
    with pytest.raises(executor_module.InformationSetCompilationStopped) as stopped:
        compile_observed_decision_v1(rules, normalized, prefix, {"kind": "bid"}, request,
            maximum_nodes=maximum_nodes, maximum_depth=8, maximum_worlds=3, maximum_seconds=30, native_cp_sat=True)
    assert limits == [maximum_nodes, maximum_nodes - 3]
    assert stopped.value.reason == "maximum_nodes"
    assert stopped.value.partial_receipt["evidence"]["compiledWorldNodes"] == 6


@pytest.mark.parametrize("passes", [0, 7])
@pytest.mark.parametrize("evaluator", ["current_raw_point_margin_v1", "auction_hand_margin_v1"])
def test_native_bid_quality_proof_and_append_boundary(root, rules, passes, evaluator):
    source, request = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"} for _ in range(passes)],
               escalationActions=[], projectActions=[], projectDeclarations=[],
               projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize(prefix, decision_point={"kind": "bid"})
    request_raw = request.to_dict()
    request_raw.pop("sha256")
    request_raw.update(targetTeam=Seat.parse(normalized.next_actor).team.name.lower(),
                       selectedAction="pass", horizon={"unit": "plies", "value": 1}, boundaryEvaluator=evaluator)
    request = ScenarioProofRequestV1.build(request_raw)
    compilation = compile_observed_decision_v1(
        rules, normalized, prefix, {"kind": "bid"}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, native_cp_sat=True,
    )
    proof = executor_module.prove_actor_information_set_v1(
        compilation, request, maximum_nodes=1000, maximum_worlds=1,
        maximum_seconds=30, native_action_quality=True,
    )
    assert proof.completion_status == "bounded_complete"
    quality = proof.model_document()["actionQuality"]
    assert quality["actualAction"] == proof.selected_action == "pass"
    assert quality["rootActorRelation"] == "target"
    # A one-ply raw-score boundary has no bidding signal: legal bids all tie.
    # This fixture proves the integration boundary, not expert bidding strength.
    if evaluator == "current_raw_point_margin_v1":
        assert set(quality["optimalActions"]) == {row["code"] for row in normalized.legal_actions}
    else:
        assert set(quality["optimalActions"]) < {row["code"] for row in normalized.legal_actions}
        assert compilation.graph.model_document["nativeAuctionModels"]
        evaluation = compilation.graph.model_document["boundaryEvaluation"]
        assert evaluation["kind"] == "heuristic" and evaluation["scale"] > 0
        assert evaluation["profileHash"] == canonical_sha256(rules.data["bidding"]["heuristics"])
        assert compilation.graph.analysis_scope["boundaryEvaluation"] == evaluation
    next_prefix = normalizer.append_normalized_action(prefix, normalized, proof.selected_action)
    after = normalizer.normalize(next_prefix, decision_point={"kind": "bid"})
    assert len(next_prefix.auction_actions) == passes + 1
    if passes == 7:
        assert after.state["allPassed"] is True
        assert not after.legal_actions and after.next_actor is None
    else:
        assert after.next_actor != normalized.next_actor
    assert len(prefix.auction_actions) == passes


@pytest.mark.parametrize("selected_action", [None, "sun"])
def test_native_bid_hand_construction_accepts_a_recorded_tied_maximum(root, rules, selected_action):
    from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1

    source, request = _quality_continuation_input(root, "bounded")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    deck = [None] * 32
    for position, code in zip(layout.initial_positions[Seat.NORTH],
                              ("SA", "HA", "C10", "DK", "S8"), strict=True):
        deck[position] = Card.parse(code).id
    deck[layout.purchase_position] = Card.parse("DQ").id
    remaining = iter(index for index in range(32) if index not in deck)
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(dealer="east", deckOrder=[next(remaining) if card is None else card for card in deck],
               auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[],
               projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    request_raw = request.to_dict()
    request_raw.pop("sha256")
    request_raw.update(horizon={"unit": "plies", "value": 1},
                       boundaryEvaluator="auction_hand_margin_v1")
    if selected_action is not None:
        request_raw["selectedAction"] = selected_action
    request = ScenarioProofRequestV1.build(request_raw)
    compilation = compile_observed_decision_v1(
        rules, normalized, prefix, {"kind": "bid"}, request,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, native_cp_sat=True,
    )
    proof = executor_module.prove_actor_information_set_v1(
        compilation, request, maximum_nodes=1000, maximum_worlds=1,
        maximum_seconds=30, native_action_quality=True,
    )
    assert proof.completion_status == "bounded_complete"
    quality = proof.model_document()["actionQuality"]
    assert set(quality["optimalActions"]) == {"pass", "sun"}
    assert quality["actualAction"] == (selected_action or "pass") and quality["regret"] == 0
    assert quality["actualActionOptimal"] is True
    if selected_action is None:
        assert quality["selectionTieBreak"] == "stable_action_id"

    model = OrToolsScenarioModelV1(rules=rules, dealer=Seat.EAST)
    model.add_native_auction_fragment(required_actions=[{"action": "sun"}],
        required_actions_complete=True, require_optimal_actions=True)
    for variable, card in zip(model.card_at_position, prefix.deck_order, strict=True):
        model.model.add(variable == card)
    result = model.solve(maximum_seconds=5, search_branching="automatic")
    assert result.status == "satisfied"
    assert result.auction.completion_actions[0].action == "sun"


@pytest.mark.parametrize("codes,expected", [
    (("SJ", "S9", "SA", "S10", "SK"), "hokum:spades"),
    (("S7", "H7", "C7", "D7", "S8"), "pass"),
])
@pytest.mark.parametrize("audit", [False, True])
def test_native_auction_continuation_selects_hand_based_action(root, rules, codes, expected, audit):
    source, request = _quality_continuation_input(root, "bounded")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    deck = [None] * 32
    for position, code in zip(layout.initial_positions[Seat.NORTH], codes, strict=True):
        deck[position] = Card.parse(code).id
    deck[layout.purchase_position] = Card.parse("SQ").id
    remaining = iter(index for index in range(32) if index not in deck)
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(dealer="east", deckOrder=[next(remaining) if card is None else card for card in deck],
               auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[],
               projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    request_raw = request.to_dict()
    request_raw.pop("sha256")
    request_raw.update(horizon={"unit": "plies", "value": 1},
                       boundaryEvaluator="auction_hand_margin_v1")
    recorded_raw = prefix.to_dict()
    recorded_raw.pop("sha256")
    recorded_raw["auctionActions"] = [{"action": "pass"} for _ in range(8)]
    recorded = ExactTracePrefixV1.build(recorded_raw) if audit else None
    result = executor_module.resolve_quality_auction_continuation_v1(
        rules, prefix, ScenarioProofRequestV1.build(request_raw),
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=8,
        maximum_seconds=30, maximum_decision_seconds=15,
        maximum_evidence_bytes=10_000_000,
        recorded_trace=recorded,
    )
    assert result["status"] == ("quality_failed" if audit else "complete"), result["partialReceipt"]
    row = result["decisions"][0]
    assert row["actor"] == "north" and row["action"] == ("pass" if audit else expected)
    assert row["actualActionOptimal"] is (not audit or expected == "pass")
    assert row["optimalActionCount"] == 1
    assert row["selectionReason"] == ("recorded_action" if audit else "native_actor_information_set_best_value")
    if audit:
        assert result["finalPrefix"] == recorded.to_dict()
    assert row["sourcePrefixHash"] == prefix.sha256 and not prefix.auction_actions
    artifact = result["artifacts"][row["proofArtifactHash"]]
    assert canonical_sha256(artifact) == row["proofArtifactHash"]
    assert artifact["request"]["mode"] == "bounded"
    assert artifact["graph"]["hiddenWorlds"]["count"] == 1
    assert artifact["graph"]["hiddenWorlds"]["complete"] is False
    assert artifact["allocationModel"]["boundaryEvaluation"]["kind"] == "heuristic"
    assert artifact["allocationModel"]["nativeAuctionModels"]


def _assert_quality_ownership_rejects_mutations(result, prefix, rules, policy):
    actor = Seat.parse(result["decisions"][0]["actor"])
    for field, replacement in (("action", "substituted"),
                               ("actor", Seat((int(actor) + 2) % 4).name.lower()),
                               ("regret", -1), ("proofArtifactHash", "0" * 64),
                               ("nextExactStateId", "substituted")):
        changed = copy.deepcopy(result)
        row = changed["decisions"][0]
        row[field] = replacement
        row["sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="quality"):
            executor_module.validate_native_decision_quality_evidence_v1(
                changed, source_prefix=prefix, rules_hash=rules.sha256,
                request=policy.request_for("play", "north"), policy=policy)


def test_explicit_native_budget_routes_play_without_finite_graph(root, rules, monkeypatch):
    prefix, request = _quality_continuation_input(root, "bounded")
    budget = _quality_budget_profile(root, maximumHiddenWorlds=1, maximumCacheBytes=30_000_000,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1",
        lambda *args, **kwargs: pytest.fail("native budget entered finite graph compiler"))
    result = resolve_quality_play_continuation_v1(rules, prefix, request,
        proof_budget_profiles={"test": budget}, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=1, maximum_seconds=60, maximum_decision_seconds=20,
        maximum_evidence_bytes=30_000_000)
    assert result["status"] == "complete", result.get("partialReceipt")
    assert result["decisions"]
    for item in result["artifacts"].values():
        kind = item["proofModel"]["kind"]
        assert "nativeExecution" in item or kind in {
            "forced_action_optimality_model_v1", "native_bounded_play_quality_model_v1"}
        if kind == "native_bounded_play_quality_model_v1":
            graph = item["graph"]
            assert graph["kind"] == "native_observed_play_selection_execution_v1"
            assert graph["sha256"] == canonical_sha256({key: value for key, value in graph.items() if key != "sha256"})
            assert item["proof"]["completionScope"] == "selected_action_supplied_worlds_bounded"


@pytest.mark.real_clock
@pytest.mark.parametrize("real_solver", [False, True, "second", "third", "last", "last_deep", "last_deep_offline", "continuous", "full_round", "full_hokum"])
def test_explicit_native_budget_routes_project_without_finite_graph(root, monkeypatch, real_solver, record_property):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1)
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    if real_solver == "full_hokum":
        raw["auctionActions"] = [{"action": action} for action in ("hokum", "pass", "pass", "pass")]
    if real_solver in {"second", "third"}:
        count = 1 if real_solver == "second" else 2
        raw["projectActions"] = raw["projectActions"][:count]
        raw["playActions"] = raw["playActions"][:count]
    elif real_solver not in {"last", "last_deep", "last_deep_offline"}:
        raw["projectActions"] = []
        raw["playActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    _, request = _quality_continuation_input(root, "bounded")
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": request.utility, "budgetProfile": "test",
            "horizon": {"unit": "tricks", "value": 2 if real_solver in {"last_deep", "last_deep_offline"} else 1},
            "boundaryEvaluator": request.boundary_evaluator} for kind in SCENARIO_DECISION_KINDS}})
    budget = _quality_budget_profile(root, maximumWallSeconds=180, maximumCpuSeconds=180,
        maximumCacheBytes=30_000_000, nativeObserved={"schemaVersion": 1,
        "maximumIterations": 1000, "maximumQueries": 2000, "maximumObservations": 2000,
        "maximumBoundPrefixes": 2000, "maximumCachedPrefixesPerBranch": 2000})
    reached = []
    def project(self, normalized, current, requested, **kwargs):
        assert normalized.decision_kind == "project" and current == prefix
        assert kwargs["budget"].native_observed is not None
        reached.append(normalized.next_actor)
        raise InformationSetCompilationStopped("cancelled", stage="native_observed_project")
    if not real_solver:
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_observed_project", project)
    if real_solver in {"full_round", "full_hokum"}:
        original_play = OrToolsActorInformationSetCompilerV1._prove_observed_play
        def observed_play(self, normalized, *args, **kwargs):
            try:
                return original_play(self, normalized, *args, **kwargs)
            except ValueError as exc:
                raise AssertionError({"error": str(exc), "trick": normalized.state["currentTrick"]["index"],
                    "projects": normalized.state["projects"],
                    "winner": normalized.state.get("projectWinningTeam")}) from exc
        monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_prove_observed_play", observed_play)
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1",
        lambda *args, **kwargs: pytest.fail("native project entered finite graph compiler"))
    result = resolve_quality_play_continuation_v1(rules, prefix, policy,
        proof_budget_profiles={"test": budget}, maximum_nodes=1000, maximum_depth=8,
        maximum_worlds=1, maximum_seconds=(
            300 if real_solver in {"full_round", "full_hokum", "last_deep_offline"}
            else 90 if real_solver == "continuous" else 30
        ),
        # Full-round green p99 is 35.5s on the reference machine; 1.40x slow-family ratio
        # and 3x margin give 149.1s, rounded to the 180s declared budget.
        maximum_decision_seconds=60 if real_solver == "last_deep_offline" else 180 if real_solver in {"full_round", "full_hokum"} else 20,
        maximum_evidence_bytes=30_000_000,
        checkpoint_callback=lambda row: reached.append(row) if real_solver and row["kind"] == "quality_decision_checkpoint_v1" else None,
        cancel_check=lambda: real_solver not in {"continuous", "full_round", "full_hokum"} and real_solver and bool(reached))
    if real_solver in {"continuous", "full_round", "full_hokum"}:
        partial = result.get("partialReceipt") or {}
        execution_receipt = result.get("executionReceipt") or {}
        assert result["status"] == "complete", json.dumps({"stage": partial.get("stage"),
            "reason": partial.get("exhaustionReason"), "decisions": len(result["decisions"]),
            "evidence": {key: value for key, value in partial.get("evidence", {}).items()
                if key in {"retainedBytes", "candidateBytes", "maximumBytes", "sectionBytes"}},
            "artifactSectionBytes": [{key: len(json.dumps(value, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8"))
                for key, value in artifact.items()} for artifact in result["artifacts"].values()],
            "elapsedSeconds": execution_receipt.get("elapsedSeconds"),
            "lastDecisionTimings": execution_receipt.get("decisionTimings", [])[-3:]}, sort_keys=True)
        assert len(result["finalPrefix"]["playActions"]) == 32
        from baloot_platform.research.ortools_nonplay_graphs import decode_native_model_receipt_v1
        from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
        validate_native_decision_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, request=policy.request_for("play", "north"), policy=policy, rules=rules)
        assert execution_receipt["evidenceBytes"] <= 30_000_000
        model_count, model_bytes = 0, 0
        for artifact in result["artifacts"].values():
            if "nativeExecution" not in artifact:
                continue
            preparations = ([item["preparation"] for item in artifact["graph"]["preparations"].values()]
                if artifact["allocationModel"].get("reference") == "graph.preparations"
                else [artifact["allocationModel"]])
            for preparation in preparations:
                restored_bytes = 0
                for model in preparation["models"]:
                    restored = decode_native_model_receipt_v1(model["modelReceipt"],
                        maximum_model_bytes=preparation["maximumModelBytes"] - restored_bytes)
                    assert restored["sourceModelHash"] == model["modelHash"]
                    restored_bytes += len(restored["modelIdentity"]["model"].encode("utf-8"))
                    model_count += 1
                assert restored_bytes == preparation["modelBytes"]
                model_bytes += restored_bytes
        record_property("nativeContinuationEvidence", json.dumps({"case": real_solver,
            "completionScope": "bounded_project_play_continuation",
            "decisions": len(result["decisions"]), "plays": 32, "restoredModels": model_count,
            "uncompressedModelBytes": model_bytes, "retainedEvidenceBytes": execution_receipt["evidenceBytes"],
            "maximumEvidenceBytes": 30_000_000, "elapsedSeconds": execution_receipt["elapsedSeconds"]}, sort_keys=True))
        return
    if real_solver:
        partial = result.get("partialReceipt") or {}
        receipt, stages = partial, []
        while receipt:
            evidence = receipt.get("evidence", {})
            usage = evidence.get("resources", {}).get("usage", {})
            stages.append({"stage": receipt.get("stage"), "reason": receipt.get("exhaustionReason"),
                "usage": {key: usage[key] for key in ("masterIterations", "counterexampleQueries", "observations") if key in usage}})
            receipt = evidence.get("causeReceipt", {})
        assert result["decisions"] and result["decisions"][0]["decisionKind"] == "project", {
            "stages": stages}
        assert result["decisions"][0]["actualActionOptimal"] is True
        assert all("nativeExecution" in row for row in result["artifacts"].values())
        from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
        tampered = copy.deepcopy(result)
        old_hash, artifact = next(iter(tampered["artifacts"].items()))
        artifact["nativeExecution"]["budgetHash"] = "0" * 64
        new_hash = canonical_sha256(artifact)
        tampered["artifacts"] = {new_hash: artifact}
        for row in tampered["decisions"]:
            if row["proofArtifactHash"] == old_hash:
                row["proofArtifactHash"] = new_hash
                row["sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
        tampered["sha256"] = canonical_sha256({key: value for key, value in tampered.items()
            if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="native execution reference checksum"):
            validate_native_decision_quality_evidence_v1(tampered, source_prefix=prefix,
                rules_hash=rules.sha256, request=policy.request_for("play", "north"), policy=policy)
    else:
        assert reached == ["north"]
    assert result["status"] != "complete"

    if not real_solver:
        # An exact paired optimization must not fix optional declarations.
        early = _optional_project_prefix_v1()
        early.update(projectActions=early["projectActions"][:1], playActions=[])
        early = ExactTracePrefixV1.build(early)
        profile_id = "offline_heavyweight_v1"
        profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
            encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"][profile_id]
        exact_policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
            "visibility": "complete_state", "decisions": {kind: {
                "utility": "awarded_point_margin", "budgetProfile": profile_id}
                for kind in SCENARIO_DECISION_KINDS}})
        routed = []

        def full_root(_rules, normalized, current, decision, request, **kwargs):
            assert normalized.decision_kind == "play" and current == early
            assert request.visibility == "complete_state" and kwargs["budget"].profile_id == profile_id
            assert 0 < kwargs["maximum_seconds"] <= 30
            routed.append(current.sha256)
            raise InformationSetCompilationStopped("cancelled", stage="native_complete_state_play")

        with monkeypatch.context() as patch:
            patch.setattr(executor_module, "_complete_state_quality_artifact_v1", full_root)
            routed_result = resolve_quality_play_continuation_v1(rules, early, exact_policy,
                execution_budget=ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
                    "maximumWallSeconds": 30, "maximumEvidenceBytes": 30_000_000,
                    "maximumEscalationDecisions": 20, "completeStatePlayPolicyPair": True}),
                proof_budget_profiles={profile_id: profile}, maximum_nodes=100_000, maximum_depth=35,
                maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=30,
                maximum_evidence_bytes=30_000_000)
        assert routed == [early.sha256]
        assert routed_result["status"] == "incomplete" and routed_result["decisions"] == []
        preflight = routed_result["executionReceipt"]["decisionTimings"][0]["pairedPreflight"]
        assert preflight["exhaustionReason"] == "paired_project_branch_choices"
        evidence = preflight["evidence"]
        assert evidence["preparedModels"] == [] and evidence["queriesUsed"] == evidence["iterationsUsed"] == 0
        coverage = evidence["projectCoverage"]
        assert len(coverage["sequences"]) > 1 and coverage["queries"][-1]["status"] == "infeasible"
        assert coverage["scope"]["prefixHash"] == early.sha256


def _quality_budget_profile(root, **overrides):
    profile = yaml.safe_load((root / "configs/challenge/progressive_mining_ortools_v3.yaml").read_text(encoding="utf-8"))
    return {**profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"], **overrides}


@pytest.mark.real_clock
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("audit", [False, True])
def test_native_auction_quality_continuation_retains_bounded_evidence(root, rules, cancelled, audit):
    from baloot_platform.research.ortools_nonplay_graphs import decode_native_auction_model_receipt_v1
    source, request = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"} for _ in range(7)],
               escalationActions=[], projectActions=[], projectDeclarations=[],
               projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    recorded_raw = prefix.to_dict()
    recorded_raw.pop("sha256")
    recorded_raw["auctionActions"].append({"action": "pass"})
    recorded = ExactTracePrefixV1.build(recorded_raw) if audit else None
    request_raw = request.to_dict()
    request_raw.pop("sha256")
    request_raw["horizon"] = {"unit": "plies", "value": 1}
    checkpoints = []
    result = executor_module.resolve_quality_auction_continuation_v1(
        rules, prefix, ScenarioProofRequestV1.build(request_raw),
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=15,
        maximum_evidence_bytes=10_000_000, cancel_check=lambda: cancelled,
        recorded_trace=recorded,
        checkpoint_callback=checkpoints.append,
    )
    assert result["scope"] == "auction_suffix_only"
    assert result["coveredFromBidIndex"] == 7
    assert result["sourcePrefixHash"] == prefix.sha256
    assert result["operation"] == ("audit_recorded" if audit else "generate")
    assert len(checkpoints) == 2 * len(result["decisions"])
    if cancelled:
        assert result["status"] == "incomplete"
        assert result["decisions"] == []
        assert result["finalPrefix"] == prefix.to_dict()
        return
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["coveredToBidIndex"] == 8
    assert len(result["decisions"]) == 1
    row = result["decisions"][0]
    assert checkpoints[1]["decision"] == row
    assert canonical_sha256(checkpoints[0]["artifact"]) == row["proofArtifactHash"]
    assert row["decisionKind"] == "bid" and row["bidIndex"] == 7
    assert row["actualActionOptimal"] is True
    artifact = result["artifacts"][row["proofArtifactHash"]]
    assert artifact["request"]["mode"] == "bounded"
    assert artifact["proofModel"]["actionQuality"]["sha256"] == row["actionQualityHash"]
    models = artifact["allocationModel"]["nativeAuctionModels"]
    assert models and all(world["requestHash"] == artifact["request"]["sha256"] for world in models)
    for world in models:
        assert world["sourcePrefixHash"] == prefix.sha256
        assert len(world["prefixHash"]) == len(world["exactHash"]) == 64
        assert world["models"]
        for encoded in world["models"]:
            receipt = decode_native_auction_model_receipt_v1(encoded, maximum_model_bytes=10_000_000)
            assert receipt["modelHash"] == canonical_sha256(receipt["modelIdentity"])
            assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
    assert canonical_sha256(artifact) == row["proofArtifactHash"]
    final = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(result["finalPrefix"]), decision_point={"kind": "bid"})
    assert final.exact_state_id == result["finalExactStateId"]
    assert not final.legal_actions
    if audit:
        assert result["finalPrefix"] == recorded.to_dict()
        assert row["action"] == "pass" and row["selectionReason"] == "recorded_action"
        assert final.state["allPassed"] is True
    for field, replacement in (("action", "sun"), ("actor", "north"),
                               ("actor", Seat((int(Seat.parse(row["actor"])) + 2) % 4).name.lower()),
                               ("regret", -1), ("optimalActionCount", 0),
                               ("proofArtifactHash", "0" * 64),
                               ("nextExactStateId", "substituted")):
        changed = copy.deepcopy(result)
        changed_row = changed["decisions"][0]
        if changed_row[field] == replacement:
            replacement = "south"
        changed_row[field] = replacement
        changed_row["sha256"] = canonical_sha256({key: value for key, value in changed_row.items() if key != "sha256"})
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match=("decision quality optimal action count"
                if field == "optimalActionCount" else "auction quality")):
            executor_module.validate_native_auction_quality_evidence_v1(
                changed, source_prefix=prefix, rules_hash=rules.sha256,
                request=ScenarioProofRequestV1.build(request_raw))
    for incomplete_flag in (False, None, 1):
        changed = copy.deepcopy(result)
        old_hash = changed["decisions"][0]["proofArtifactHash"]
        changed_artifact = changed["artifacts"].pop(old_hash)
        model = changed_artifact["proofModel"]
        quality = model["actionQuality"]
        quality["optimalActionsComplete"] = incomplete_flag
        quality["sha256"] = canonical_sha256({**{key: value for key, value in quality.items() if key != "sha256"},
            "modelProto": model.get("modelProto")})
        changed_artifact["proof"]["identities"]["modelHash"] = canonical_sha256(model)
        new_hash = canonical_sha256(changed_artifact)
        changed["artifacts"][new_hash] = changed_artifact
        changed_row = changed["decisions"][0]
        changed_row.update(proofArtifactHash=new_hash, actionQualityHash=quality["sha256"])
        changed_row["sha256"] = canonical_sha256({key: value for key, value in changed_row.items() if key != "sha256"})
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="partial optimal actions"):
            executor_module.validate_native_auction_quality_evidence_v1(
                changed, source_prefix=prefix, rules_hash=rules.sha256,
                request=ScenarioProofRequestV1.build(request_raw))
    for mutation in ("deal", "action"):
        changed = copy.deepcopy(result)
        final_raw = changed["finalPrefix"]
        final_raw.pop("sha256")
        if mutation == "deal":
            final_raw["deckOrder"][0], final_raw["deckOrder"][1] = final_raw["deckOrder"][1], final_raw["deckOrder"][0]
        else:
            final_raw["auctionActions"][-1] = {"action": "sun"}
        changed["finalPrefix"] = ExactTracePrefixV1.build(final_raw).to_dict()
        changed_row = changed["decisions"][-1]
        changed_row["nextPrefixHash"] = changed["finalPrefix"]["sha256"]
        changed_row["sha256"] = canonical_sha256({key: value for key, value in changed_row.items() if key != "sha256"})
        changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="auction quality"):
            executor_module.validate_native_auction_quality_evidence_v1(
                changed, source_prefix=prefix, rules_hash=rules.sha256,
                request=ScenarioProofRequestV1.build(request_raw))


@pytest.mark.parametrize("mutation", ["actor", "deal", "missing", "extra", "assertion", "proof", "partial_optimal_actions"])
def test_native_auction_quality_rejects_invalid_recorded_trace(root, rules, mutation, monkeypatch):
    source, request = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[{"action": "pass"} for _ in range(7)],
               escalationActions=[], projectActions=[], projectDeclarations=[],
               projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    recorded = copy.deepcopy(raw)
    recorded["auctionActions"].append({"action": "pass"})
    if mutation == "actor":
        recorded["auctionActions"][-1]["actor"] = rules.structural_domain.dealer_right(Seat.parse(prefix.dealer)).name.lower()
    elif mutation == "deal":
        recorded["deckOrder"][0], recorded["deckOrder"][1] = recorded["deckOrder"][1], recorded["deckOrder"][0]
    elif mutation == "missing":
        recorded["auctionActions"].pop()
    elif mutation == "extra":
        recorded["auctionActions"].append({"action": "pass"})
    elif mutation == "assertion":
        recorded["stateAssertions"] = {"buyer": "north"}
    else:
        original = executor_module.prove_actor_information_set_v1
        def substituted(*args, **kwargs):
            proof = original(*args, **kwargs)
            document = copy.deepcopy(proof.model_document())
            other = proof.selected_action
            if mutation == "partial_optimal_actions":
                document["actionQuality"]["optimalActionsComplete"] = False
            else:
                other = next(action for action in document["actionQuality"]["optimalActions"] if action != "pass")
                document["actionQuality"]["actualAction"] = other
            return SimpleNamespace(selected_action=other, completion_status=proof.completion_status,
                                   model_document=lambda: document, to_dict=proof.to_dict)
        monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", substituted)
    request_raw = request.to_dict()
    request_raw.pop("sha256")
    request_raw["horizon"] = {"unit": "plies", "value": 1}
    def run():
        return executor_module.resolve_quality_auction_continuation_v1(
            rules, prefix, ScenarioProofRequestV1.build(request_raw),
            recorded_trace=ExactTracePrefixV1.build(recorded),
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000)
    if mutation in {"assertion", "extra", "proof", "partial_optimal_actions"}:
        result = run()
        assert result["status"] == "model_error"
        if mutation in {"proof", "partial_optimal_actions"}:
            assert result["partialReceipt"]["reason"] == "action_quality_binding_disagreement"
            assert result["finalPrefix"] == prefix.to_dict()
    else:
        with pytest.raises(ValueError):
            run()
    assert len(prefix.auction_actions) == 7


@pytest.mark.real_clock
@pytest.mark.parametrize("through_backend", [False, True])
def test_native_round_quality_audits_all_pass_without_post_auction(root, rules, monkeypatch, through_backend, tmp_path):
    source, _ = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[], escalationActions=[], projectActions=[],
               projectDeclarations=[], projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    recorded = copy.deepcopy(raw)
    recorded["auctionActions"] = [{"action": "pass"} for _ in range(8)]
    recorded = ExactTracePrefixV1.build(recorded)
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "late_game_exact_v1",
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in ("bid", "escalation", "project", "play")}})
    def forbidden(*args, **kwargs):
        pytest.fail("all-pass round entered post-auction quality")
    native_post_auction = executor_module.resolve_quality_post_auction_continuation_v1
    monkeypatch.setattr(executor_module, "resolve_quality_post_auction_continuation_v1", forbidden)
    if through_backend:
        from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2

        profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
        profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"].update(
            maximumNodes=1000, maximumDepth=8, maximumHiddenWorlds=1,
            maximumWallSeconds=15, maximumCpuSeconds=15)
        backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity={
            "bundleRevisionId": "CBR-QUALITY-TEST", "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64})
        original_trace = {"completeExactTracePrefix": recorded.to_dict()}
        attached = backend.audit_quality_trace(
            rules, original_trace, prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
            maximum_seconds=60, recorded_events=[])
        result = attached["quality"]
        assert "playingQualityEvidence" not in original_trace
        assert attached["trace"]["completeExactTracePrefix"] == original_trace["completeExactTracePrefix"]
        assert "executionReceipt" not in attached["trace"]["playingQualityEvidence"]["receipt"]
        with pytest.raises(ValueError, match="already has immutable"):
            backend.audit_quality_trace(rules, attached["trace"], prefix, {}, policy, recorded_events=[])
        with pytest.raises(ValueError, match="requires recorded"):
            backend.audit_quality_trace(rules, original_trace, prefix, {}, policy, recorded_events=None)
        generated = backend.generate_quality_trace(rules, prefix,
            {"kind": "escalation", "selector": {}}, policy, cancel_check=lambda: True)
        assert generated["quality"]["status"] == "incomplete"
        generated_receipt = generated["trace"]["playingQualityEvidence"]["receipt"]
        assert generated_receipt["operation"] == "generate"
        assert generated_receipt["recordedPrefixHash"] is None
        assert generated["trace"]["completeExactTracePrefix"] == prefix.to_dict()
        assert "executionReceipt" not in generated_receipt
        monkeypatch.setattr(executor_module, "resolve_quality_post_auction_continuation_v1", native_post_auction)
        generated_profile = copy.deepcopy(profile)
        generated_profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"]["maximumHiddenWorlds"] = 2
        generated_profile["scenarioResearchV4"]["qualityExecution"].update(
            earlyPlayPolicyId="observer_safe_trick_control_v1",
            exactPlayFromTrick=7,
        )
        generated_backend = OrToolsScenarioResearchBackendV2(challenge_profile=generated_profile, bundle_identity={
            "bundleRevisionId": "CBR-QUALITY-GENERATED-TWO-WORLDS", "semanticBundleHash": "c" * 64,
            "effectiveRuntimeConfigHash": "d" * 64})
        generated_rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
        generated_complete = generated_backend.generate_quality_trace(generated_rules, prefix,
            {"kind": "escalation", "selector": {}}, policy, maximum_seconds=300)
        generated_evidence = tmp_path / "generated-quality-result.json"
        generated_evidence.write_text(json.dumps(generated_complete, sort_keys=True), encoding="utf-8")
        assert generated_complete["quality"]["status"] == "complete", str(generated_evidence)
        generated_round = generated_complete["trace"]["playingQualityEvidence"]["receipt"]
        assert generated_round["operation"] == "generate"
        assert len(generated_round["auction"]["decisions"]) == len(generated_round["finalPrefix"]["auctionActions"])
        assert len(generated_round["finalPrefix"]["playActions"]) == (
            0 if generated_round["auction"]["auctionOutcome"]["allPassed"] else 32)
        assert all(row["actualActionOptimal"] for row in generated_round["auction"]["decisions"])
        quality_executor = OrToolsUnifiedResearchExecutorV1(
            generated_rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-QUALITY-GENERATED-TWO-WORLDS", bundle_hash="c" * 64,
            quality_policy=policy, quality_execution=generated_backend.authority.quality_execution,
            quality_proof_profiles=generated_profile["scenarioResearchV4"]["proofBudgets"]["profiles"],
        )
        candidate = quality_executor.candidate_from_quality_trace(generated_complete["trace"], maximum_seconds=30)
        terminal = OrToolsExactStateNormalizerV1(generated_rules, native_cp_sat=True).normalize(
            ExactTracePrefixV1.build(generated_round["finalPrefix"]),
            decision_point={"kind": "play", "selector": {}}, maximum_seconds=30,
        )
        def retained_generation(*args, **kwargs):
            assert args[1] == prefix
            assert args[3].to_dict() == policy.to_dict()
            assert kwargs["execution_budget"] == generated_backend.authority.quality_execution
            assert kwargs["proof_budget_profiles"]["late_game_exact_v1"]["maximumHiddenWorlds"] == 2
            kwargs["final_normalized_callback"](terminal)
            return {**generated_round, "executionReceipt": generated_complete["quality"]["executionReceipt"]}
        with monkeypatch.context() as generation_patch:
            generation_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", retained_generation)
            regenerated = quality_executor.generate_quality_candidate(candidate, maximum_seconds=30)
            generation_patch.setattr(
                executor_module, "resolve_quality_round_continuation_v1",
                lambda *args, **kwargs: {**generated_round,
                    "executionReceipt": generated_complete["quality"]["executionReceipt"]},
            )
            with pytest.raises(ValueError, match="did not retain one terminal state"):
                quality_executor.generate_quality_candidate(candidate, maximum_seconds=30)
        assert regenerated.status == "feasible"
        assert regenerated.graph is None and regenerated.proof_outcome is None
        assert regenerated.candidate.sha256 == candidate.sha256
        assert regenerated.candidate.trace == candidate.trace
        assert "_finalNormalized" not in json.dumps(generated_round, sort_keys=True)
        assert candidate.origin == "search"
        assert candidate.trace["completeExactTracePrefix"] == generated_round["finalPrefix"]
        assert candidate.trace["playingQualityEvidence"] == generated_complete["trace"]["playingQualityEvidence"]
        assert len(candidate.trace["completeExactTracePrefix"]["playActions"]) == 32
        selected_scenario = _scenario(
            constraints=_leaf("PRED-CONTRACT", "auction.contract", {"contract": candidate.trace["contract"]}),
            outcome=_leaf("PRED-WINNER", "outcome.team_round", {"team": "north_south", "status": "won"}),
        )
        selected_prefix = executor_module._search_decision_prefix(
            candidate, selected_scenario, native_cp_sat=True,
        )
        assert len(selected_prefix.play_actions) == 24
        assert selected_prefix.round_number == prefix.round_number
        assert selected_prefix.match_score == prefix.match_score
        selected_state = OrToolsExactStateNormalizerV1(generated_rules, native_cp_sat=True).normalize(
            selected_prefix, decision_point=selected_scenario.decision_point, maximum_seconds=30,
        )
        assert selected_state.legal_actions
        from baloot_platform.research.ortools_tail_first_candidate import OrToolsTailFirstCandidateCompilerV1
        qualification_scenario = _scenario(
            constraints=selected_scenario.constraints.to_dict(),
            outcome=_leaf("PRED-WINNER", "outcome.team_round", {
                "team": candidate.trace["score"]["roundWinner"], "status": "won",
            }),
        )
        with monkeypatch.context() as campaign_patch:
            campaign_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", retained_generation)
            def no_rewrite(*args, **kwargs):
                pytest.fail("an audited full round must not rematerialize its continuation")
            campaign_patch.setattr(executor_module, "_materialize_play_policy_continuation_v1", no_rewrite)
            admission_checks = []
            original_validation = executor_module.validate_trace_quality_evidence_v1
            def checked_admission(*args, **kwargs):
                admission_checks.append(kwargs)
                return original_validation(*args, **kwargs)
            campaign_patch.setattr(executor_module, "validate_trace_quality_evidence_v1", checked_admission)
            campaign_patch.setattr(GenericCompleteTraceCompilerV1, "__call__",
                                  lambda *args, **kwargs: OrToolsCandidateAttemptV1("feasible", candidate))
            campaign_patch.setattr(OrToolsTailFirstCandidateCompilerV1, "__call__",
                                  lambda *args, **kwargs: OrToolsCandidateAttemptV1("feasible", candidate))
            qualified = quality_executor.execute(
                qualification_scenario, _proof(target_team=candidate.trace["score"]["roundWinner"]),
                maximum_seconds=60, maximum_proof_nodes=10000, maximum_proof_depth=12,
                maximum_prefix_lift_attempts=1,
            )
            assert len(admission_checks) == 1
            assert admission_checks[0]["rules"] is generated_rules
            assert admission_checks[0]["required_policy"] == quality_executor.quality_policy
            assert admission_checks[0]["required_execution"] == quality_executor.quality_execution
            assert admission_checks[0]["required_proof_profiles"] == quality_executor.quality_proof_profiles
            assert admission_checks[0]["required_search_profiles"] == quality_executor.quality_search_profiles
            exclusions = []
            def retained_candidate(_compiler, _scenario, excluded, **kwargs):
                exclusions.append(excluded)
                return OrToolsCandidateAttemptV1("feasible", candidate)
            campaign_patch.setattr(GenericCompleteTraceCompilerV1, "__call__", retained_candidate)
            campaign_patch.setattr(OrToolsTailFirstCandidateCompilerV1, "__call__", retained_candidate)
            campaign_patch.setattr(quality_executor, "generate_quality_candidate", lambda *args, **kwargs:
                                  OrToolsCandidateAttemptV1("not_found", None, reason="quality_generation_target_miss"))
            missed = quality_executor.execute_campaign(
                qualification_scenario, _proof(target_team=candidate.trace["score"]["roundWinner"]),
                result_count=1, symmetry_policy_id="rules_preserving_v1",
                allow_symmetry_only_variants=False, minimum_decision_state_distance=4,
                maximum_candidates=2, maximum_seconds=60, maximum_campaign_seconds=60,
                maximum_proof_nodes=10000, maximum_proof_depth=12, maximum_prefix_lift_attempts=1,
            )
            assert len(missed.executions) == 2
            assert missed.executions[0].status == "semantic_postcondition_miss"
            assert missed.executions[0].execution_receipt["completedCandidates"] == 1
            assert candidate.no_good_vector in exclusions[1]
        assert qualified.status == "qualified"
        assert qualified.proof["completionStatus"] == "exact_complete"
        qualified_documents = qualified.semantic_artifact_documents()
        assert qualified.research_authority["traceHash"] == canonical_sha256(
            qualified_documents["trace"]
        )
        assert (
            qualified.research_authority["proofModelHash"]
            == qualified.proof["identities"]["modelHash"]
        )
        assert qualified.candidate.trace["completeExactTracePrefix"] == generated_round["finalPrefix"]
        assert qualified.candidate.trace["playingQualityEvidence"] == candidate.trace["playingQualityEvidence"]
        with pytest.raises(executor_module._PolicyContinuationCompilerDisagreement,
                           match="audited full rounds cannot be rematerialized"):
            executor_module._materialize_play_policy_continuation_v1(
                rules=generated_rules, candidate=candidate, source_normalization=selected_state,
                graph=None, proof=None, request=_proof(), maximum_seconds=1, native_cp_sat=True,
            )
        tampered = copy.deepcopy(dict(candidate.trace))
        tampered["completeExactTracePrefix"] = selected_prefix.to_dict()
        with pytest.raises(ValueError, match="trace quality final prefix"):
            quality_executor.candidate_from_quality_trace(tampered, maximum_seconds=30)
        assert result["qualityExecution"] == backend.authority.quality_execution.to_dict()
        assert result["auction"]["proofBudgetProfiles"]["late_game_exact_v1"]["maximumHiddenWorlds"] == 1
    else:
        result = executor_module.resolve_quality_round_continuation_v1(
        rules, prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
        opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=60,
        maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
        recorded_trace=recorded, recorded_events=[])
    assert result["status"] == "complete", result["partialReceipt"]
    assert result["scope"] == "all_decisions" and result["postAuction"] is None
    if not through_backend:
        shorter_runtime = executor_module.resolve_quality_round_continuation_v1(
            rules, prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
            opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=60,
            maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
            runtime_maximum_seconds=30, recorded_trace=recorded, recorded_events=[],
        )
        assert shorter_runtime["sha256"] == result["sha256"]
    assert result["finalPrefix"] == recorded.to_dict()
    assert len(result["auction"]["decisions"]) == 8
    assert all(row["selectionReason"] == "recorded_action" for row in result["auction"]["decisions"])
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items()
                                               if key not in {"sha256", "executionReceipt"}})
    assert prefix.auction_actions == ()
    from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
    receipt = copy.deepcopy(result)
    if through_backend:
        receipt["sha256"] = receipt.pop("sourceQualityHash")
        receipt.pop("backendIdentity")
        receipt.pop("configurationAuthority")
    kwargs = dict(source_prefix=prefix, rules_hash=rules.sha256, policy=policy,
                  decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
    validate_native_round_quality_evidence_v1(receipt, **kwargs)
    for changed_rows in (receipt["auction"]["decisions"][:-1],
                         [receipt["auction"]["decisions"][0]] * len(receipt["auction"]["decisions"])):
        changed = copy.deepcopy(receipt)
        changed["auction"]["decisions"] = changed_rows
        for envelope in (changed["auction"], changed):
            envelope["sha256"] = canonical_sha256({key: value for key, value in envelope.items()
                if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="quality decision (coverage|chain) mismatch"):
            validate_native_round_quality_evidence_v1(changed, **kwargs)
    from baloot_platform.research.result_authority import validate_trace_quality_evidence_v1
    trace = {"completeExactTracePrefix": receipt["finalPrefix"], "playingQualityEvidence": {
        "sourcePrefix": prefix.to_dict(), "decisionPoint": kwargs["decision_point"], "receipt": receipt}}
    validate_trace_quality_evidence_v1(trace, rules_hash=rules.sha256)
    validate_trace_quality_evidence_v1(trace, rules_hash=rules.sha256, required_policy=policy)
    with pytest.raises(ValueError, match="proof budgets differ from frozen profiles"):
        validate_trace_quality_evidence_v1(trace, rules_hash=rules.sha256,
            required_policy=policy,
            required_proof_profiles={row["budgetProfile"]: {"substituted": True}
                                     for row in policy.decisions.values()})
    different_policy = policy.to_dict()
    with pytest.raises(ValueError, match="execution budget differs from frozen budget"):
        validate_trace_quality_evidence_v1(trace, rules_hash=rules.sha256,
            required_policy=policy, required_execution=ScenarioQualityExecutionBudgetV1.build({
                "schemaVersion": 1, "maximumWallSeconds": 1,
                "maximumEvidenceBytes": 1, "maximumEscalationDecisions": 1}))
    different_policy.pop("sha256")
    different_policy["decisions"]["play"]["budgetProfile"] = "different_budget"
    with pytest.raises(ValueError, match="differs from requested policy"):
        validate_trace_quality_evidence_v1(trace, rules_hash=rules.sha256,
            required_policy=ScenarioPlayingQualityPolicyV1.build(different_policy))
    with pytest.raises(ValueError, match="requires quality evidence"):
        validate_trace_quality_evidence_v1({}, rules_hash=rules.sha256, required_policy=policy)
    from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
    artifact_store = LocalContentAddressedArtifactStore(tmp_path / "quality-artifacts")
    encoded = json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    reference = artifact_store.put(encoded, media_type="application/vnd.balootx.ortools-complete-trace-v1+json",
                                   name="quality-trace.json")
    reopened = LocalContentAddressedArtifactStore(tmp_path / "quality-artifacts")
    loaded = json.loads(reopened.get(reference["uri"]))
    assert loaded == trace
    validate_trace_quality_evidence_v1(loaded, rules_hash=rules.sha256)
    cancelled = executor_module.resolve_quality_round_continuation_v1(
        rules, prefix, kwargs["decision_point"], policy,
        opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=60,
        maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
        recorded_trace=recorded, recorded_events=[], cancel_check=lambda: True)
    assert cancelled["status"] == "incomplete"
    assert cancelled["finalPrefix"] == prefix.to_dict()
    partial_trace = copy.deepcopy(trace)
    partial_trace["playingQualityEvidence"]["receipt"] = cancelled
    validate_trace_quality_evidence_v1(partial_trace, rules_hash=rules.sha256)
    with pytest.raises(ValueError, match="requires complete quality evidence"):
        validate_trace_quality_evidence_v1(partial_trace, rules_hash=rules.sha256,
            required_policy=policy)
    assert partial_trace["completeExactTracePrefix"] == recorded.to_dict()
    substituted = {**trace, "completeExactTracePrefix": prefix.to_dict()}
    with pytest.raises(ValueError, match="trace quality final prefix"):
        validate_trace_quality_evidence_v1(substituted, rules_hash=rules.sha256)
    for field, replacement in (("status", "invented"), ("status", "quality_failed"),
                               ("status", "incomplete"), ("operation", "generate"), ("scope", "selected_root_only")):
        changed = copy.deepcopy(receipt)
        changed[field] = replacement
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items()
                                             if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="round quality"):
            validate_native_round_quality_evidence_v1(changed, **kwargs)
    changed = copy.deepcopy(receipt)
    changed["finalPrefix"] = prefix.to_dict()
    changed["sha256"] = canonical_sha256({k: v for k, v in changed.items()
                                         if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="round quality"):
        validate_native_round_quality_evidence_v1(changed, **kwargs)


@pytest.mark.parametrize("field,value", [("maximum_seconds", 31),
    ("maximum_evidence_bytes", 1001), ("maximum_escalation_decisions", 21)])
def test_round_quality_refuses_frozen_budget_overrun_before_solver(rules, monkeypatch, field, value):
    raw = _optional_project_prefix_v1()
    raw.update(auctionActions=[], projectActions=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "exact",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test"}
                      for kind in SCENARIO_DECISION_KINDS}})
    limits = dict(maximum_seconds=30, maximum_evidence_bytes=1000, maximum_escalation_decisions=20)
    limits[field] = value
    def forbidden(*args, **kwargs):
        pytest.fail("over-budget request entered solver")
    monkeypatch.setattr(executor_module, "resolve_quality_auction_continuation_v1", forbidden)
    with pytest.raises(ValueError, match="frozen execution budget"):
        executor_module.resolve_quality_round_continuation_v1(
            rules, prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
            opportunity_order=("west", "east", "north", "south"), maximum_nodes=1000,
            maximum_depth=8, maximum_worlds=1, maximum_decision_seconds=10,
            execution_budget=ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
                "maximumWallSeconds": 30, "maximumEvidenceBytes": 1000, "maximumEscalationDecisions": 20}),
            **limits)


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["complete", "quality_failed", "cancelled", "maximum_seconds", "maximum_evidence_bytes", "auction_incomplete"])
@pytest.mark.parametrize("frozen", [False, True])
def test_round_quality_handoff_preserves_campaign_budget(rules, monkeypatch, stop, frozen):
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=[], playActions=[])
    purchased = ExactTracePrefixV1.build(raw)
    raw["auctionActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in ("bid", "escalation", "project", "play")}})
    clock, calls, checkpoints, verified = [0], [], [], set()
    def final_normalized_callback(_value):
        return None
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def auction(*args, **kwargs):
        clock[0] = 31 if stop == "maximum_seconds" else 2
        assert (kwargs["execution_budget"] is not None) is frozen
        return {"status": "incomplete" if stop == "auction_incomplete" else "quality_failed" if stop == "quality_failed" else "complete",
            "partialReceipt": {"reason": "retained"} if stop in {"auction_incomplete", "quality_failed"} else None,
            "finalPrefix": purchased.to_dict(), "auctionOutcome": {"allPassed": False},
            "executionReceipt": {"evidenceBytes": 6}}
    def post(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["_verified_artifacts"] is verified
        assert kwargs["_curated_focal_precheck"] is True
        assert kwargs["final_normalized_callback"] is final_normalized_callback
        assert args[1].sha256 == purchased.sha256
        assert kwargs["maximum_seconds"] == 30 and kwargs["runtime_maximum_seconds"] == 28
        assert kwargs["maximum_evidence_bytes"] == 994
        return {"status": "complete", "partialReceipt": None, "finalPrefix": purchased.to_dict(),
                "executionReceipt": {"evidenceBytes": 1}}
    monkeypatch.setattr(executor_module, "resolve_quality_auction_continuation_v1", auction)
    monkeypatch.setattr(executor_module, "resolve_quality_post_auction_continuation_v1", post)
    result = executor_module.resolve_quality_round_continuation_v1(
        rules, prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
        opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=30,
        maximum_decision_seconds=10, maximum_evidence_bytes=6 if stop == "maximum_evidence_bytes" else 1000,
        cancel_check=lambda: stop == "cancelled",
        checkpoint_callback=checkpoints.append,
        _verified_artifacts=verified,
        _curated_focal_precheck=True,
        final_normalized_callback=final_normalized_callback,
        execution_budget=ScenarioQualityExecutionBudgetV1.build({
            "schemaVersion": 1, "maximumWallSeconds": 30,
            "maximumEvidenceBytes": 1000, "maximumEscalationDecisions": 20}) if frozen else None)
    assert ("qualityExecution" in result) is frozen
    if frozen:
        assert result["qualityExecution"]["maximumWallSeconds"] == 30
    assert len(calls) == (1 if stop in {"complete", "quality_failed"} else 0)
    assert result["status"] == (stop if calls else "incomplete")
    if stop == "quality_failed":
        assert result["partialReceipt"] == {"reason": "retained"}
    assert result["executionReceipt"]["evidenceBytes"] == (7 if calls else 6)
    if stop in {"cancelled", "maximum_seconds", "maximum_evidence_bytes"}:
        assert result["partialReceipt"]["exhaustionReason"] == stop
    assert result["finalPrefix"] == purchased.to_dict()
    assert checkpoints == [{"schemaVersion": 1, "kind": "quality_stage_checkpoint_v1",
        "rulesHash": rules.sha256, "stage": "auction",
        "elapsedSeconds": 31 if stop == "maximum_seconds" else 2,
        "executionReceipt": {"evidenceBytes": 6}}]


@pytest.mark.real_clock
@pytest.mark.parametrize("completed_post_decisions,recorded_start,evaluator", [
    (count, start, "current_raw_point_margin_v1")
    for count in (0, 1) for start in (False, True, "pass")
] + [(0, False, "auction_hand_margin_v1"), (None, False, "auction_hand_margin_v1"),
     ("trick", False, "auction_hand_margin_v1")])
def test_round_quality_native_purchase_reaches_post_auction_analysis(root, monkeypatch, completed_post_decisions, recorded_start, evaluator, record_property, request):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    full_round = completed_post_decisions in (None, "trick")
    measurements = {}
    if completed_post_decisions == "trick":
        import time
        from baloot_platform.research.ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
        def measure(owner, method):
            original_method = getattr(owner, method)
            measurement = {"calls": 0, "inclusiveSeconds": 0.0}
            measurements[f"{owner.__name__}.{method}"] = measurement
            def timed(*args, **kwargs):
                started = time.perf_counter()
                measurement["calls"] += 1
                stage = None
                if method == "compile_graph_worlds":
                    from collections import Counter
                    measurement.setdefault("worldGraphs", []).append([
                        {"nodes": len(world.graph.nodes),
                         "outDegrees": dict(sorted(Counter(len(node.edges) for node in world.graph.nodes).items())),
                         "observations": len(set(world.observation_ids.values())),
                         "decisionKind": world.decision_kind}
                        for world in args[1]])
                if owner is ortools_exact_state.cp_model.CpSolver and method == "solve":
                    variables = args[1].proto.variables
                    model_kind = variables[0].name if variables else "empty"
                    stage = measurement.setdefault("modelCalls", {}).setdefault(
                        model_kind, {"calls": 0, "inclusiveSeconds": 0.0})
                    stage["calls"] += 1
                if owner is OrToolsExactStateNormalizerV1 and method == "normalize":
                    prefix_value = args[1]
                    prefix_hash = (prefix_value.sha256 if isinstance(prefix_value, ExactTracePrefixV1)
                                   else canonical_sha256(prefix_value))
                    counts = measurement.setdefault("prefixCalls", {})
                    counts[prefix_hash] = counts.get(prefix_hash, 0) + 1
                    if isinstance(prefix_value, ExactTracePrefixV1):
                        key = f"{kwargs.get('decision_point', {}).get('kind')}:{len(prefix_value.play_actions)}:{len(prefix_value.project_actions)}"
                        stage = measurement.setdefault("stageCalls", {}).setdefault(
                            key, {"calls": 0, "inclusiveSeconds": 0.0, "legalActionCounts": {}},
                        )
                        stage["calls"] += 1
                try:
                    result = original_method(*args, **kwargs)
                    if stage is not None and "legalActionCounts" in stage:
                        count = str(len(result.legal_actions))
                        histogram = stage["legalActionCounts"]
                        histogram[count] = histogram.get(count, 0) + 1
                    return result
                finally:
                    elapsed = time.perf_counter() - started
                    measurement["inclusiveSeconds"] += elapsed
                    if stage is not None:
                        stage["inclusiveSeconds"] += elapsed
                    if method == "_enumerate_hidden_groups":
                        exact = kwargs.get("exact")
                        measurement.setdefault("allocations", []).append({
                            "seconds": elapsed, "poolSize": len(args[1]),
                            "groupSizes": list(args[2]),
                            "stateId": exact.exact_state_id if exact else None,
                            "decisionKind": exact.decision_kind if exact else "bid",
                            "nextActor": exact.next_actor if exact else None,
                        })
            monkeypatch.setattr(owner, method, timed)
        measure(OrToolsExactStateNormalizerV1, "normalize")
        measure(OrToolsCompleteStateGraphCompilerV1, "compile")
        from baloot_platform.research import ortools_exact_state
        from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
        measure(OrToolsActorInformationSetCompilerV1, "_enumerate_hidden_groups")
        measure(OrToolsActorInformationSetCompilerV1, "compile_graph_worlds")
        measure(ortools_exact_state, "_cp_model_hash")
        measure(ortools_exact_state, "_native_fixed_play_prefix")
        measure(ortools_exact_state, "_native_role_projection")
        measure(ortools_exact_state, "_native_project_prefix_projection")
        measure(ortools_exact_state, "_native_project_legal_actions")
        measure(OrToolsPlayTailStageV1, "build_model")
        measure(ortools_exact_state, "attach_project_lifecycle_fragment_v1")
        measure(ortools_exact_state.cp_model.CpSolver, "solve")
        request.addfinalizer(lambda: record_property(
            "qualityCallMeasurements", json.dumps(measurements, sort_keys=True)))
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    if evaluator == "auction_hand_margin_v1":
        from baloot_platform.core.dealing import DealLayoutV1
        layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
        deck = [None] * 32
        for position, code in zip(layout.initial_positions[Seat.NORTH], ("SJ", "S9", "SA", "S10", "SK"), strict=True):
            deck[position] = Card.parse(code).id
        deck[layout.purchase_position] = Card.parse("SQ").id
        remaining = iter(card for card in range(32) if card not in deck)
        raw["deckOrder"] = [next(remaining) if card is None else card for card in deck]
    raw.update(auctionActions=[], projectActions=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    submitted = ExactTracePrefixV1.build({**raw, "auctionActions": [{"action": "pass"}]}) if recorded_start == "pass" else prefix
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "tricks" if completed_post_decisions == "trick" and kind == "play" else "plies", "value": 1}, "boundaryEvaluator": evaluator if kind == "bid" else "current_raw_point_margin_v1"}
            for kind in ("bid", "escalation", "project", "play")}})
    original = executor_module.compile_observed_decision_v1
    calls = []
    def stop_at_post_auction(*args, **kwargs):
        calls.append(args[3]["kind"])
        if not full_round and sum(kind != "bid" for kind in calls) > completed_post_decisions:
            raise InformationSetCompilationStopped("cancelled", stage="handoff_test_boundary")
        return original(*args, **kwargs)
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", stop_at_post_auction)
    execution_budget = ScenarioQualityExecutionBudgetV1.build({
        "schemaVersion": 1, "maximumWallSeconds": 300,
        "maximumEvidenceBytes": 30_000_000, "maximumEscalationDecisions": 20,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1", "exactPlayFromTrick": 7,
    }) if completed_post_decisions == "trick" else None
    decision_point = {
        "kind": "escalation",
        "selector": {} if recorded_start == "pass" else {"actingSeat": "west"},
    }
    result = executor_module.resolve_quality_round_continuation_v1(
        rules, prefix, decision_point, policy,
        recorded_trace=submitted if recorded_start else None,
        recorded_events=[] if recorded_start else None,
        continue_recorded_prefix=bool(recorded_start),
        opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=2,
        maximum_seconds=300 if full_round else 90,
        maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000 if full_round else 10_000_000,
        execution_budget=execution_budget)
    assert result["auction"]["status"] in {"complete", "quality_failed"}, result["auction"]["partialReceipt"]
    assert result["auction"]["auctionOutcome"]["contract"] is not None
    if full_round:
        from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
        assert result["status"] == "complete", json.dumps(result["partialReceipt"], sort_keys=True)
        assert set(calls) == {"bid", "escalation", "project", "play"}
        assert len(result["finalPrefix"]["playActions"]) == 32
        # Forced plays use their constant-size CP-SAT certificate and therefore
        # do not enter the expensive observed-decision compiler.
        assert 0 < calls.count("play") <= (8 if completed_post_decisions == "trick" else 32)
        for row in result["auction"]["decisions"]:
            artifact = result["auction"]["artifacts"][row["proofArtifactHash"]]
            assert artifact["allocationModel"]["boundaryEvaluation"]["kind"] == "heuristic"
            assert artifact["request"]["boundaryEvaluator"] == evaluator
        validate_native_round_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, policy=policy,
            decision_point=decision_point)
        for kind in ("project", "play"):
            changed = copy.deepcopy(result)
            changed_play = changed["postAuction"]["projectPlay"]
            index = next(i for i, row in reversed(list(enumerate(changed_play["decisions"])))
                         if row.get("decisionKind", "play") == kind)
            changed_play["decisions"].pop(index)
            changed_play["sha256"] = canonical_sha256({key: value for key, value in changed_play.items()
                if key not in {"sha256", "executionReceipt"}})
            changed["postAuction"]["sha256"] = canonical_sha256({key: value for key, value in changed["postAuction"].items()
                if key not in {"sha256", "executionReceipt"}})
            changed["sha256"] = canonical_sha256({key: value for key, value in changed.items()
                if key not in {"sha256", "executionReceipt"}})
            with pytest.raises(ValueError, match="quality decision coverage mismatch"):
                validate_native_round_quality_evidence_v1(changed, source_prefix=prefix,
                    rules_hash=rules.sha256, policy=policy, decision_point=decision_point)
        return
    assert calls[0] == "bid" and calls[-1] in {"escalation", "project"}
    assert result["status"] == "incomplete" and result["postAuction"] is not None
    assert result["partialReceipt"]["exhaustionReason"] == "cancelled"
    assert len(result["auction"]["decisions"]) == calls.count("bid")
    assert sum(kind != "bid" for kind in calls) == completed_post_decisions + 1
    retained = len(result["postAuction"]["escalation"]["decisions"])
    if result["postAuction"]["projectPlay"] is not None:
        retained += len(result["postAuction"]["projectPlay"]["decisions"])
    assert retained == completed_post_decisions
    assert result["postAuction"]["sourcePrefixHash"] == result["auction"]["finalPrefix"]["sha256"]
    assert prefix.auction_actions == ()
    if evaluator == "auction_hand_margin_v1":
        from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
        for row in result["auction"]["decisions"]:
            artifact = result["auction"]["artifacts"][row["proofArtifactHash"]]
            assert artifact["allocationModel"]["boundaryEvaluation"]["kind"] == "heuristic"
            assert artifact["request"]["boundaryEvaluator"] == evaluator
        validate_native_round_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, policy=policy,
            decision_point=decision_point)
    if recorded_start:
        from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
        assert result["auction"]["operation"] == "audit_then_generate"
        assert result["postAuction"]["operation"] == "generate"
        assert result["submittedPrefix"] == submitted.to_dict()
        if recorded_start == "pass":
            assert result["auction"]["decisions"][0]["selectionReason"] == "recorded_action"
            assert result["auction"]["finalPrefix"]["auctionActions"][0] == {"action": "pass"}
        validate_native_round_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, policy=policy,
            decision_point=decision_point)


@pytest.mark.parametrize("invalid", ["missing", "workers", "wall", "cpu", "shape"])
def test_native_quality_rejects_invalid_resolved_budgets_before_normalization(root, rules, monkeypatch, invalid):
    prefix, request = _quality_continuation_input(root, "bounded")
    profiles = {"test": _quality_budget_profile(root)}
    if invalid == "missing":
        profiles = {}
    elif invalid == "workers":
        profiles["test"]["maximumWorkers"] = 2
    elif invalid == "wall":
        profiles["test"]["maximumWallSeconds"] = float("nan")
    elif invalid == "cpu":
        profiles["test"]["maximumCpuSeconds"] = float("inf")
    else:
        del profiles["test"]["maximumNodes"]

    def forbidden(*args, **kwargs):
        pytest.fail("invalid budget reached normalization")

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", forbidden)
    with pytest.raises(ValueError):
        resolve_quality_play_continuation_v1(
            rules, prefix, request, proof_budget_profiles=profiles,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
        )


@pytest.mark.real_clock
def test_native_quality_reports_cpu_budget_exhaustion(root, rules, monkeypatch):
    prefix, request = _quality_continuation_input(root, "bounded")
    counter = itertools.count()
    monkeypatch.setattr(executor_module.time, "process_time", lambda: float(next(counter)))

    def forbidden(*args, **kwargs):
        pytest.fail("exhausted CPU allowance reached graph construction")

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", forbidden)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, proof_budget_profiles={"test": _quality_budget_profile(root, maximumCpuSeconds=0.5)},
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
    )
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == "maximum_cpu_seconds"
    assert result["finalPrefix"] == prefix.to_dict()
    assert result["decisions"] == []


@pytest.mark.real_clock
def test_native_quality_charges_world_worker_cpu_before_proof(root, rules, monkeypatch):
    prefix, request = _quality_continuation_input(root, "bounded")
    monkeypatch.setattr(executor_module.time, "process_time", lambda: 0.0)

    def compiled(*args, **kwargs):
        assert kwargs["graph_workers"] == 4
        kwargs["worker_cpu_callback"](1.0)

    def forbidden(*args, **kwargs):
        pytest.fail("exhausted worker CPU allowance reached proof")

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", compiled)
    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", forbidden)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, proof_budget_profiles={"test": _quality_budget_profile(root, maximumCpuSeconds=0.5)},
        graph_workers=4, maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000)
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == "maximum_cpu_seconds"
    assert result["decisions"] == [] and result["finalPrefix"] == prefix.to_dict()


@pytest.mark.parametrize("limits,reason", [
    ({"maximumNodes": 1}, "maximum_nodes"),
    ({"maximumDepth": 2}, "maximum_depth"),
    ({"maximumHorizonActions": 2}, "maximum_horizon_actions"),
])
def test_native_quality_budget_refusal_precedes_graph_construction(root, rules, monkeypatch, limits, reason):
    prefix, request = _quality_continuation_input(root, "bounded")

    def forbidden(*args, **kwargs):
        pytest.fail("exhausted budget reached graph construction")

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", forbidden)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, proof_budget_profiles={"test": _quality_budget_profile(root, **limits)},
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
    )
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == reason
    assert result["finalPrefix"] == prefix.to_dict()
    assert result["decisions"] == []


@pytest.mark.real_clock
def test_quality_escalation_history_is_owned_and_missing_is_not_empty(root, rules):
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    prefix, request = _quality_continuation_input(root, "bounded")
    history = [{"actor": "west", "actionCode": "decline"}]
    kwargs = dict(proof_budget_profiles={"test": _quality_budget_profile(root)}, maximum_nodes=1000,
        maximum_depth=8, maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=15,
        maximum_evidence_bytes=10_000_000, cancel_check=lambda: True)
    result = resolve_quality_play_continuation_v1(rules, prefix, request, escalation_events=history, **kwargs)
    assert result["status"] == "incomplete"
    assert result["escalationHistoryHash"] == canonical_sha256(history)
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, request=request, rules=rules)
    for substituted in (None, [], [{"actor": "east", "actionCode": "decline"}]):
        with pytest.raises(ValueError, match="history substitution"):
            validate_native_decision_quality_evidence_v1(result, escalation_events=substituted, **context)
    validate_native_decision_quality_evidence_v1(result, escalation_events=history, **context)
    legacy = resolve_quality_play_continuation_v1(rules, prefix, request, **kwargs)
    assert "escalationHistoryHash" not in legacy
    validate_native_decision_quality_evidence_v1(legacy, escalation_events=history, **context)
    with pytest.raises(ValueError, match="canonical actor"):
        resolve_quality_play_continuation_v1(rules, prefix, request,
            escalation_events=[{"actor": "west", "actionCode": "decline", "hidden": True}], **kwargs)


@pytest.mark.parametrize("horizon", [1, 3, 4])
def test_preplay_escalation_boundary_skips_only_irrelevant_hidden_deals(root, rules, monkeypatch, horizon):
    from baloot_platform.research.result_authority import validate_native_escalation_quality_evidence_v1

    source, base_request = _quality_continuation_input(root, "bounded")
    raw = source.to_dict()
    raw.pop("sha256")
    raw.update(dealer="east", deckOrder=list(range(32)),
        auctionActions=[{"action": action} for action in ("hokum", "pass", "pass", "pass")],
        escalationActions=[], projectActions=[], projectDeclarations=[], projectAssertions=[],
        playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "escalation", "selector": {"actingSeat": "west"}}
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision, maximum_seconds=15)
    request_raw = base_request.to_dict()
    request_raw.pop("sha256")
    request_raw.pop("selectedAction", None)
    request_raw.update(targetTeam="west_east", horizon={"unit": "plies", "value": horizon})
    request = ScenarioProofRequestV1.build(request_raw)
    order = ["west", "east", "north", "south"]

    def no_deal_enumeration(*args, **kwargs):
        raise AssertionError("hidden-deal enumeration")

    monkeypatch.setattr(executor_module.OrToolsActorInformationSetCompilerV1,
        "enumerate_graph_world_inputs", no_deal_enumeration)
    if horizon != 1:
        with pytest.raises(AssertionError, match="hidden-deal enumeration"):
            compile_observed_decision_v1(rules, normalized, prefix, decision, request,
                maximum_nodes=2000, maximum_depth=8, maximum_worlds=64,
                maximum_seconds=30, native_cp_sat=True, private_recall={}, opportunity_order=order)
        return
    result = executor_module.resolve_quality_escalation_continuation_v1(
        rules, prefix, decision, request, opportunity_order=order,
        maximum_nodes=2000, maximum_depth=8, maximum_worlds=64, maximum_seconds=30,
        maximum_decision_seconds=15, maximum_decisions=20, maximum_evidence_bytes=10_000_000,
        recorded_events=[{"actor": actor, "actionCode": "decline"} for actor in ("west", "east")],
        proof_budget_profiles={"test": _quality_budget_profile(root, maximumHiddenWorlds=64)})
    assert result["status"] == "complete", result.get("partialReceipt")
    assert len(result["decisions"]) == 2
    for artifact in result["artifacts"].values():
        assert artifact["proof"]["completionStatus"] == "bounded_complete"
        assert artifact["graph"]["hiddenWorlds"]["complete"] is False
        assert artifact["graph"]["analysisScope"]["worldCoverageComplete"] is False
        assert artifact["allocationModel"]["publicBoundaryProjection"]["cardAllocationEnumeration"] == "not_performed"
        assert artifact["allocationModel"]["escalationOrderCoverage"]["complete"] is True
        assert artifact["allocationModel"]["priorHistoryCoverage"]["complete"] is True
        assert artifact["proofModel"]["actionQuality"]["actualActionOptimal"] is True
    # Read the serialized receipt afresh, retaining private history and bounded scope.
    validate_native_escalation_quality_evidence_v1(json.loads(json.dumps(result)), source_prefix=prefix,
        rules_hash=rules.sha256, request=request, rules=rules, maximum_seconds=30)
    changed = replace(normalized, state={**normalized.state, "buyer": "south"})
    with pytest.raises(ValueError, match="identity mismatch"):
        compile_observed_decision_v1(rules, changed, prefix, decision, request,
            maximum_nodes=2000, maximum_depth=8, maximum_worlds=64,
            maximum_seconds=30, native_cp_sat=True, private_recall={}, opportunity_order=order)


def test_observed_quality_dispatch_cancels_before_constructing_models(root, rules, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )

    prefix, request = _quality_continuation_input(root, "bounded")
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"}, maximum_seconds=15,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled request constructed a compiler")

    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "__init__", forbidden)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compile_observed_decision_v1(
            rules, normalized, prefix, {"kind": "play"}, request,
            maximum_nodes=100, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=2, native_cp_sat=True, cancel_check=lambda: True,
        )
    assert stopped.value.reason == "cancelled"
    assert stopped.value.partial_receipt["stage"] == "observed_decision"


@pytest.mark.real_clock
@pytest.mark.parametrize("mode", ["exact", "bounded", "tail_exact"])
def test_native_quality_generates_new_plays_from_each_actors_own_information(root, rules, mode, monkeypatch):
    prefix, request = _quality_continuation_input(root, "bounded" if mode == "tail_exact" else mode)
    analysis = {key: value for key, value in request.to_dict().items()
                if key in {"utility", "budgetProfile", "horizon", "boundaryEvaluator"}}
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted" if mode in {"bounded", "tail_exact"} else "exact",
        "decisions": {kind: copy.deepcopy(analysis) for kind in SCENARIO_DECISION_KINDS},
    })
    preserved = copy.deepcopy(prefix.to_dict())
    original = executor_module.compile_observed_decision_v1

    watchers = []
    def without_unneeded_watchers(*args, **kwargs):
        watchers.append(kwargs["cancel_check"])
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", without_unneeded_watchers)
    checkpoints = []
    execution_budget = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 180, "maximumEvidenceBytes": 20_000_000, "maximumEscalationDecisions": 20,
        "exactPlayFromTrick": 7, "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        "exactTailProof": True}) if mode == "tail_exact" else None
    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=64 if mode in {"exact", "tail_exact"} else 1, maximum_seconds=180,
        maximum_decision_seconds=60, maximum_evidence_bytes=20_000_000,
        checkpoint_callback=checkpoints.append, execution_budget=execution_budget,
    )
    assert watchers[0] is None  # Generation needs no watchdog without a frozen CPU budget.
    if mode in {"exact", "tail_exact"}:
        assert any(callable(watcher) for watcher in watchers)  # Fresh readback is bounded independently.
    assert result["status"] == "complete", result.get("partialReceipt")
    assert result["playingQuality"] == policy.to_dict()
    assert prefix.to_dict() == preserved
    assert result["scope"] == "play_suffix_only"
    assert result["coveredFromPlayIndex"] == 27
    assert result["coveredToPlayIndex"] == 32
    assert len(result["decisions"]) == 5
    assert len(checkpoints) == 10
    for index, (checkpoint, decision) in enumerate(zip(
        checkpoints[1::2], result["decisions"], strict=True,
    )):
        assert checkpoint["decision"] == decision
        assert checkpoint["sourcePrefix"]["sha256"] == decision["sourcePrefixHash"]
        assert checkpoint["nextPrefix"]["sha256"] == decision["nextPrefixHash"]
        assert checkpoint["artifactHash"] == decision["proofArtifactHash"]
        assert checkpoint["rulesHash"] == rules.sha256
        assert checkpoint["timing"]["decisionIndex"] == index
        assert checkpoint["timing"]["decisionKind"] == checkpoint["stage"]
        assert checkpoint["timing"]["selectedActionNormalizationSeconds"] >= 0
    for checkpoint, decision in zip(checkpoints[::2], result["decisions"], strict=True):
        assert canonical_sha256(checkpoint["artifact"]) == decision["proofArtifactHash"]
    first = result["decisions"][0]
    assert first["action"] == "S7"
    _assert_quality_ownership_rejects_mutations(result, prefix, rules, policy)
    assert first["optimalActionCount"] == (2 if mode in {"exact", "tail_exact"} else 1)
    assert first["selectionTieBreak"] == ("stable_action_id" if mode in {"exact", "tail_exact"} else None)
    assert result["finalPrefix"]["playActions"][:27] == preserved["playActions"]
    final = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        result["finalPrefix"], decision_point={"kind": "play"}, maximum_seconds=10,
    )
    assert final.phase == "scored"
    assert final.exact_state_id == result["finalExactStateId"]
    source_hash = prefix.sha256
    for row in result["decisions"]:
        assert row["sourcePrefixHash"] == source_hash
        assert row["targetTeam"] == Seat.parse(row["actor"]).team.name.lower()
        artifact = result["artifacts"][row["proofArtifactHash"]]
        assert row["proofArtifactHash"] == canonical_sha256(artifact)
        assert artifact["request"]["targetTeam"] == row["targetTeam"]
        if mode == "tail_exact":
            assert artifact["request"]["mode"] == "exact"
            assert "horizon" not in artifact["request"]
            assert artifact["proof"]["completionStatus"] == "exact_complete"
            assert artifact["proofModel"]["kind"] in {
                "actor_information_set_cegis_model_v1", "forced_action_optimality_model_v1"}
        quality = artifact["proofModel"]["actionQuality"]
        assert quality["actualAction"] == row["action"]
        assert quality["regret"] == 0
        assert quality["actualActionOptimal"] is True
        assert quality["rootActorRelation"] == "target"
        assert quality["sha256"] == row["actionQualityHash"]
        assert row["optimalActionCount"] == len(quality["optimalActions"])
        source_hash = row["nextPrefixHash"]
    assert result["sha256"] == canonical_sha256({
        key: value for key, value in result.items() if key not in {"sha256", "executionReceipt"}
    })
    if mode == "tail_exact":
        from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
        validate_native_decision_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, request=policy.request_for("play", "north"),
            policy=policy, rules=rules, execution_budget=execution_budget, maximum_seconds=60)
        with pytest.raises(ValueError, match="decision request mismatch"):
            validate_native_decision_quality_evidence_v1(result, source_prefix=prefix,
                rules_hash=rules.sha256, request=policy.request_for("play", "north"),
                policy=policy, rules=rules, maximum_seconds=60)


@pytest.mark.parametrize("reason", ["cancelled", "evidence", "runtime"])
def test_native_quality_continuation_preserves_prefix_when_incomplete(root, rules, reason, monkeypatch):
    prefix, request = _quality_continuation_input(root, "bounded")
    if reason == "runtime":
        set_monotonic(monkeypatch, executor_module, 0, 0, 2)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=10,
        maximum_evidence_bytes=1 if reason == "evidence" else 20_000_000,
        cancel_check=(lambda: True) if reason == "cancelled" else None,
        runtime_maximum_seconds=1 if reason == "runtime" else None,
    )
    assert result["status"] == "incomplete"
    assert result["decisions"] == []
    assert result["finalPrefix"] == prefix.to_dict()
    assert result["partialReceipt"]["exhaustionReason"] == (
        {"cancelled": "cancelled", "evidence": "maximum_evidence_bytes", "runtime": "maximum_seconds"}[reason]
    )
    assert result["limits"]["maximumSeconds"] == 30


@pytest.mark.parametrize("stop_at", ["proof", "decision"])
def test_native_decision_checkpoint_survives_campaign_cancel_and_storage(root, rules, tmp_path, stop_at):
    import threading
    from baloot_platform.research.campaign_coordinator import (
        CampaignBudgetV1, CampaignShardOutcomeV1, DeterministicCpSatCampaignCoordinatorV1,
    )
    from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
    prefix, request = _quality_continuation_input(root, "bounded")
    cancelled, finished = threading.Event(), threading.Event()
    live = []
    def solver(shard):
        try:
            def checkpoint(value):
                shard.report_checkpoint(value)
                if value["kind"] == f"quality_{stop_at}_checkpoint_v1":
                    cancelled.set()
            result = resolve_quality_play_continuation_v1(
                rules, prefix, request, maximum_nodes=100_000, maximum_depth=8,
                maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=20,
                maximum_evidence_bytes=20_000_000,
                checkpoint_callback=checkpoint,
                cancel_check=lambda: cancelled.is_set() or shard.cancellation_requested())
            assert result["status"] == "incomplete"
            return CampaignShardOutcomeV1((), partial_evidence=shard.progress_snapshot())
        finally:
            finished.set()
    shards = partition_research_shards(run_code="RUN-CHECKPOINT01",
        run_hash=canonical_sha256("checkpoint-run"), recipe_hash=prefix.sha256,
        seed_start=0, seed_count=1, seeds_per_shard=1, maximum_attempts=1)
    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards, solver=solver,
        budget=CampaignBudgetV1(result_count=1, shard_worker_count=1,
            maximum_wall_time_seconds=30, cancellation_poll_milliseconds=5),
        cancellation_event=cancelled, progress_callback=live.append)
    finished.wait()
    assert not receipt.accepted_variants
    checkpoints = receipt.attempts[0].partial_evidence["qualityCheckpoints"]
    assert len(checkpoints) == (1 if stop_at == "proof" else 2)
    for identity, value in checkpoints.items():
        assert canonical_sha256(value) == identity
    checkpoint = next(value for value in checkpoints.values()
        if value["kind"] == "quality_proof_checkpoint_v1")
    assert checkpoint["sourcePrefix"] == prefix.to_dict()
    assert canonical_sha256(checkpoint["artifact"]) == checkpoint["artifactHash"]
    assert checkpoint["artifact"]["proof"]["completionStatus"] == "bounded_complete"
    if stop_at == "decision":
        decision = next(value for value in checkpoints.values()
            if value["kind"] == "quality_decision_checkpoint_v1")
        assert len(decision["nextPrefix"]["playActions"]) == 28
        assert decision["artifactHash"] == checkpoint["artifactHash"]
        timing = decision["timing"]
        assert 0 <= timing["checkpointCallbackSeconds"] <= timing["elapsedSeconds"]
        assert timing["elapsedSeconds"] >= timing["selectedActionNormalizationSeconds"]
    assert all("qualityCheckpoints" not in value for value in live)
    store = LocalContentAddressedArtifactStore(tmp_path / "checkpoint-artifacts")
    reference = store.put(json.dumps(receipt.to_dict(), sort_keys=True).encode("utf-8"),
        media_type="application/json", name="checkpoint-receipt.json")
    reopened = LocalContentAddressedArtifactStore(tmp_path / "checkpoint-artifacts")
    loaded = json.loads(reopened.get(reference["uri"]))
    assert loaded["attempts"][0]["partialEvidence"]["qualityCheckpoints"] == checkpoints


@pytest.mark.parametrize("kind", ["play", "interleaved_play", "project"])
def test_persisted_recall_binds_only_each_actors_own_observation(root, rules, kind):
    from baloot_platform.research.ortools_nonplay_graphs import bind_escalation_recall_v1
    _prefix, request = _quality_continuation_input(root, "bounded")
    raw = request.to_dict()
    raw.pop("sha256")
    raw["horizon"] = {"unit": "plies", "value": 1 if kind == "play" else 2}
    request = ScenarioProofRequestV1.build(raw)
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    value = _optional_project_prefix_v1()
    value["auctionActions"] = [{"action": action} for action in ("hokum", "pass", "pass", "pass")]
    value["escalationActions"] = [{"actor": "east", "action": "double", "mode": "open"}]
    value["projectActions"] = [{"actor": seat, "action": "pass"}
                               for seat in (("north", "west") if kind != "project" else ("north",))]
    if kind == "play":
        value.pop("projectActions")  # Legacy exact input exercises the non-interleaved play compiler.
    value["playActions"] = [{"actor": "north", "card": "S7", "ekka": False}]
    prefix = ExactTracePrefixV1.build(value)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=20)
    actor = normalized.next_actor
    assert actor == "west"
    def compile(recall, seconds=30):
        decision = {"kind": "play" if kind == "interleaved_play" else kind,
                    **({"selector": {"owner": actor}} if kind == "project" else {})}
        return compile_observed_decision_v1(rules, normalized, prefix, decision, request,
                maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=seconds,
                declared_maximum_seconds=30,
            native_cp_sat=True, private_recall=recall).graph
    base = compile({})
    window = canonical_sha256({"actions": []})
    own = {actor: [window]}
    bound = compile(own)
    changed = compile({**own, "north": ["b" * 64]}, seconds=25)
    key = "observerInformationHash" if kind == "play" else "rootObservationId"
    base_root = base.hidden_worlds["coverage"][key]
    assert bound.hidden_worlds["coverage"][key] == bind_escalation_recall_v1(base_root, [window])
    assert changed.hidden_worlds["coverage"][key] == bound.hidden_worlds["coverage"][key]
    assert bound.to_dict() == changed.to_dict()
    assert bound.model_document == changed.model_document
    assert bound.model_document["priorHistoryCoverage"]["complete"] is True
    assert bound.model_document["priorHistoryCoverage"]["worldCount"] == 1
    assert base.hidden_worlds["complete"] == bound.hidden_worlds["complete"] == changed.hidden_worlds["complete"]
    decisions = []
    def compare(before, after):
        assert (before.actor, before.utility) == (after.actor, after.utility)
        for (actor, observation, action), (other, bound_observation, bound_action) in zip(
                before.forced_prefix, after.forced_prefix, strict=True):
            assert (actor, action) == (other, bound_action)
            decisions.append(observation)
            if bound_observation != observation:
                assert bound_observation == bind_escalation_recall_v1(observation, [window])
        if before.utility is None:
            decisions.append(before.observation_id)
            if after.observation_id != before.observation_id:
                assert after.observation_id == bind_escalation_recall_v1(before.observation_id, [window])
        for (action, child), (other, bound_child) in zip(before.children, after.children, strict=True):
            assert action == other
            compare(child, bound_child)
    for before, after in zip(base.problem.worlds, bound.problem.worlds, strict=True):
        for (action, child), (other, bound_child) in zip(before.roots, after.roots, strict=True):
            assert action == other
            compare(child, bound_child)
    if kind != "play":
        assert decisions  # Exercise continuation decisions, not just the visible root.


@pytest.mark.parametrize("recall", [{"South": ["a" * 64]}, {"south": "a" * 64},
    {"south": ["bad"]}, {"south": ["a" * 64, "a" * 64]}])
def test_persisted_recall_rejects_invalid_identity(root, rules, recall):
    prefix, request = _quality_continuation_input(root, "bounded")
    with pytest.raises(ValueError):
        resolve_quality_play_continuation_v1(rules, prefix, request, private_recall=recall,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=30,
            maximum_decision_seconds=10, maximum_evidence_bytes=1000)


@pytest.mark.parametrize("direct", [False, True])
def test_exact_recall_requires_certified_histories_before_card_enumeration(root, rules, monkeypatch, direct):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    prefix, request = _quality_continuation_input(root, "exact")
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=15)
    other = next(seat.name.lower() for seat in Seat if seat.name.lower() != normalized.next_actor)
    def forbidden(*args, **kwargs):
        if direct:
            raise AssertionError("uncovered private history reached hidden-world enumeration")
        raise InformationSetCompilationStopped("test_card_boundary", stage="test_card_worlds")
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_enumerate_worlds", forbidden)
    with pytest.raises(InformationSetCompilationStopped) as caught:
        limits = dict(maximum_nodes=1000, maximum_depth=8, maximum_worlds=64, maximum_seconds=30,
                      private_recall={other: ["a" * 64]})
        if direct:
            OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True).compile(normalized, request, **limits)
        else:
            compile_observed_decision_v1(rules, normalized, prefix, {"kind": "play"}, request,
                native_cp_sat=True, **limits)
    if direct:
        assert caught.value.reason == "private_history_coverage_unavailable"
    else:
        assert caught.value.reason == "test_card_boundary"
        coverage = caught.value.partial_receipt["evidence"]["priorHistoryCoverage"]
        assert coverage["complete"] is True
        assert coverage["worlds"] == [{"worldId": canonical_sha256({}), "privateRecall": {}}]


@pytest.mark.parametrize("mutation", ["checksum", "fields", "duplicate", "own", "empty", "world_limit", "coercive_world_limit"])
def test_certified_history_world_shape_is_checked_before_play_compilation(root, rules, monkeypatch, mutation):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    prefix, request = _quality_continuation_input(root, "bounded")
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix, maximum_seconds=15)
    row = {"worldId": canonical_sha256({}), "privateRecall": {}}
    worlds = [row]
    if mutation == "checksum":
        row["worldId"] = "0" * 64
    elif mutation == "fields":
        row["extra"] = True
    elif mutation in {"duplicate", "world_limit"}:
        worlds *= 2 if mutation == "duplicate" else 5
    elif mutation == "own":
        row["privateRecall"] = {normalized.next_actor: ["a" * 64]}
        row["worldId"] = canonical_sha256(row["privateRecall"])
    elif mutation == "empty":
        worlds = []
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid history worlds reached card enumeration")
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "_enumerate_worlds", forbidden)
    with pytest.raises((ValueError, InformationSetCompilationStopped)):
        OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True).compile(normalized, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=True if mutation == "coercive_world_limit" else 4, maximum_seconds=30,
            private_recall={}, prior_history_worlds=worlds)


@pytest.mark.parametrize("kind", ["play", "project"])
def test_certified_history_receipt_survives_later_compilation_exhaustion(root, rules, monkeypatch, kind):
    from baloot_platform.research.ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    prefix, request = _quality_continuation_input(root, "bounded")
    if kind == "project":
        rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
        prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix, maximum_seconds=15)
    def exhausted(*args, **kwargs):
        raise InformationSetCompilationStopped("maximum_nodes", stage="test_card_worlds", evidence={"count": 0})
    monkeypatch.setattr(OrToolsActorInformationSetCompilerV1, "compile" if kind == "play" else "compile_nonplay", exhausted)
    decision = {"kind": kind, **({"selector": {"owner": normalized.next_actor}} if kind == "project" else {})}
    with pytest.raises(InformationSetCompilationStopped) as caught:
        compile_observed_decision_v1(rules, normalized, prefix, decision, request,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=30,
            native_cp_sat=True, private_recall={})
    receipt = caught.value.partial_receipt
    assert receipt["stage"] == "test_card_worlds" and caught.value.reason == "maximum_nodes"
    assert receipt["evidence"]["priorHistoryCoverage"]["complete"] is True
    assert receipt["evidence"]["priorHistoryCoverage"]["evidence"]
    assert receipt["evidence"]["causeReceipt"]["evidence"] == {"count": 0}


@pytest.mark.parametrize("allowance,message", [
    (True, "quality readback deadline"), (0, "quality readback deadline"),
    (-1, "quality readback deadline"), (float("nan"), "quality readback deadline"),
    (float("inf"), "quality readback deadline"), (31, "runtime allowance"),
])
def test_native_quality_rejects_invalid_runtime_allowance(root, rules, allowance, message):
    prefix, request = _quality_continuation_input(root, "bounded")
    with pytest.raises(ValueError, match=message):
        resolve_quality_play_continuation_v1(rules, prefix, request, maximum_nodes=1000,
            maximum_depth=8, maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=10,
            maximum_evidence_bytes=1000, runtime_maximum_seconds=allowance)


@pytest.mark.real_clock
@pytest.mark.parametrize("whole_trace", [False, True])
def test_native_quality_audits_a_frontier_inferior_ace_without_repairing_the_line(root, rules, whole_trace, record_property):
    prefix, request = _quality_continuation_input(root, "bounded")
    recorded = [{"card": card, "ekka": False} for card in ("SA", "C9", "DA", "D8", "S7")]
    preserved = copy.deepcopy(recorded)
    recording = {"recorded_actions": recorded}
    if whole_trace:
        document = prefix.to_dict()
        document.pop("sha256")
        document["playActions"].extend(recorded)
        recording = {"recorded_trace": ExactTracePrefixV1.build(document)}
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=1, maximum_seconds=60, maximum_decision_seconds=20,
        maximum_evidence_bytes=20_000_000, **recording,
    )
    assert result["status"] == "quality_failed", result.get("partialReceipt")
    assert result["operation"] == "audit_recorded"
    assert len(result["decisions"]) == 5
    assert result["coveredToPlayIndex"] == 32
    assert result["finalPrefix"]["playActions"][27:] == preserved
    assert result["finalPrefix"]["sha256"] == result["recordedPrefixHash"]
    assert recorded == preserved
    first = result["decisions"][0]
    assert first["action"] == "SA"
    assert first["selectionReason"] == "recorded_action"
    assert first["selectionTieBreak"] is None
    assert first["optimalActionCount"] == 1
    assert first["actualActionOptimal"] is False
    assert first["regret"] > 0
    quality = result["artifacts"][first["proofArtifactHash"]]["proofModel"]["actionQuality"]
    assert quality["optimalActions"] == ["S7"]
    assert quality["analysisScope"]["horizon"] == {"unit": "tricks", "value": 1}
    assert quality["analysisScope"]["worldCoverageComplete"] is False
    timings = result["executionReceipt"]["decisionTimings"]
    assert len(timings) == len(result["decisions"])
    fields = ("compilationSeconds", "proofSeconds", "selectedActionNormalizationSeconds")
    for index, timing in enumerate(timings):
        assert timing["decisionIndex"] == index and timing["decisionKind"] == "play"
        assert all(timing[field] >= 0 for field in fields)
    totals = {field: sum(row[field] for row in timings) for field in fields}
    assert sum(totals.values()) <= result["executionReceipt"]["elapsedSeconds"]
    assert result["executionReceipt"]["initialNormalizationSeconds"] >= 0
    assert result["executionReceipt"]["checkpointCallbackSeconds"] == 0
    record_property("qualityStageSeconds", json.dumps(totals, sort_keys=True))
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items()
                                               if key not in {"sha256", "executionReceipt"}})


def test_recorded_search_quality_stops_at_first_bad_action(root, rules):
    prefix, request = _quality_continuation_input(root, "bounded")
    recorded = [{"card": card, "ekka": False} for card in ("SA", "C9", "DA", "D8", "S7")]
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=1, maximum_seconds=60, maximum_decision_seconds=20,
        maximum_evidence_bytes=20_000_000, recorded_actions=recorded,
        stop_on_recorded_quality_failure=True,
    )
    assert result["status"] == "quality_failed"
    assert result["partialReceipt"]["reason"] == "recorded_action_not_optimal"
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["actualActionOptimal"] is False
    assert len(result["finalPrefix"]["playActions"]) == len(prefix.play_actions) + 1


def test_recorded_candidate_forwards_first_failure_policy(root, rules, monkeypatch):
    recorded = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, bundle_revision_id="CBR-FIRST-FAILURE-TEST", bundle_hash="a" * 64,
    )
    marker = object()

    def capture(_candidate, **kwargs):
        assert kwargs["stop_on_recorded_quality_failure"] is True
        assert kwargs["recorded_trace"] == recorded
        return marker

    monkeypatch.setattr(executor, "_resolve_quality_candidate", capture)
    assert executor.audit_quality_candidate(
        None, recorded_trace=recorded, recorded_events=[], maximum_seconds=1,
        decision_point={"kind": "play", "selector": {}},
        stop_on_recorded_quality_failure=True,
    ) is marker


def test_candidate_identity_avoids_copies_but_exports_detached(monkeypatch):
    digest = canonical_sha256("candidate-identity")
    trace = {"actions": [{"actor": "north", "card": "HA"}]}
    state = {"hands": {"north": ["HA", "H7"]}}
    receipt = OrToolsResearchStageReceiptV1("deal", digest, digest, digest, "OPTIMAL")
    with monkeypatch.context() as patch:
        patch.setattr(executor_module.copy, "deepcopy",
            lambda value: pytest.fail("read-only candidate hashing copied evidence"))
        candidate = OrToolsNeutralCandidateV1("search", trace, state, (), {"deal": receipt})
    exported = candidate.to_dict()
    assert candidate.sha256 == canonical_sha256({
        key: value for key, value in exported.items() if key != "sha256"})
    exported["trace"]["actions"][0]["card"] = "H7"
    exported["state"]["hands"]["north"].clear()
    assert candidate.trace == trace == {"actions": [{"actor": "north", "card": "HA"}]}
    assert candidate.state == state == {"hands": {"north": ["HA", "H7"]}}


def test_v3_candidate_semantic_identity_excludes_execution_timing() -> None:
    digest = canonical_sha256("stable-stage")

    def candidate(elapsed_seconds: float) -> OrToolsNeutralCandidateV1:
        return OrToolsNeutralCandidateV1(
            "search",
            {},
            {},
            (),
            {
                "deal": OrToolsResearchStageReceiptV1(
                    "deal",
                    digest,
                    digest,
                    digest,
                    "OPTIMAL",
                    elapsed_seconds=elapsed_seconds,
                )
            },
        )

    first = executor_module._upgrade_candidate_receipts_v3_1(candidate(0.1))
    second = executor_module._upgrade_candidate_receipts_v3_1(candidate(9.9))

    assert first.sha256 == second.sha256
    assert first.stage_receipts["deal"].elapsed_seconds == 0.0
    assert second.stage_receipts["deal"].elapsed_seconds == 0.0


def _optional_project_prefix_v1(*, deck_order: list[int] | None = None) -> dict:
    return {
        "schemaVersion": 1,
        "deckOrder": list(range(32)) if deck_order is None else deck_order,
        "dealer": "east",
        "roundNumber": 1,
        "matchScore": [0, 0],
        "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [],
        "projectActions": [
            {"actor": actor, "action": "pass"}
            for actor in ("north", "west", "south")
        ],
        "projectDeclarations": [],
        "projectAssertions": [],
        "playActions": [
            {"actor": "north", "card": "S7", "ekka": False},
            {"actor": "west", "card": "S10", "ekka": False},
            {"actor": "south", "card": "SK", "ekka": False},
        ],
        "stateAssertions": {},
    }


def test_observed_quality_forwards_solver_progress_without_claiming_proof(root, monkeypatch):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    profile_id = "offline_heavyweight_v1"
    profiles = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]
    budget = ScenarioProofBudgetV1.build(profile_id, profiles[profile_id])
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: {"utility": "awarded_point_margin",
            "budgetProfile": profile_id} for kind in SCENARIO_DECISION_KINDS}})
    events = []

    def incomplete(self, *args, progress_callback, **kwargs):
        progress_callback({"kind": "quality_solver_progress_v1", "phase": "counterexample",
            "masterCount": 2, "queryCount": 3, "bindingCount": 1})
        raise executor_module.InformationSetCompilationStopped("maximum_queries", stage="test_progress")

    monkeypatch.setattr(executor_module.OrToolsActorInformationSetCompilerV1,
        "_solve_native_post_auction_policy_bounds", incomplete)
    with pytest.raises(executor_module.InformationSetCompilationStopped, match="maximum_queries"):
        executor_module._observed_play_quality_artifact_v1(rules,
            SimpleNamespace(next_actor="east"), prefix, policy.request_for("project", "east"),
            budget=budget, opportunity_order=("west", "east", "north", "south"),
            escalation_events=[], maximum_conflicts=250_000, maximum_depth=64,
            maximum_seconds=5, decision_kind="project", progress_callback=events.append)
    assert events == [
        {"kind": "quality_solver_progress_v1", "phase": "preparation",
            "masterCount": 0, "queryCount": 0, "bindingCount": 0,
            "decisionKind": "project", "proofPhase": "best"},
        {"kind": "quality_solver_progress_v1", "phase": "counterexample",
            "masterCount": 2, "queryCount": 3, "bindingCount": 1,
            "decisionKind": "project", "proofPhase": "best"},
    ]


def test_observed_project_emits_live_progress_before_proof_completion(root):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    profile_id = "offline_heavyweight_v1"
    profiles = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: {"utility": "awarded_point_margin",
            "budgetProfile": profile_id} for kind in SCENARIO_DECISION_KINDS}})
    events = []
    result = resolve_quality_play_continuation_v1(rules, prefix, policy,
        checkpoint_callback=events.append,
        cancel_check=lambda: any(row.get("kind") == "quality_solver_progress_v1" for row in events),
        maximum_nodes=100_000, maximum_depth=64, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=25, maximum_evidence_bytes=30_000_000,
        opportunity_order=("west", "east", "north", "south"), escalation_events=[],
        proof_budget_profiles=profiles, search_budget_profiles={profile_id: {"maximumConflicts": 250_000}})
    progress = [row for row in events if row.get("kind") == "quality_solver_progress_v1"]
    assert progress and not result["decisions"]
    assert progress[0]["decisionKind"] == "project"
    assert {"deckOrder", "hands", "observation"}.isdisjoint(progress[0])
    assert result["status"] == "incomplete"


@pytest.mark.real_clock
@pytest.mark.parametrize("other_actor", [False, True])
def test_optional_project_pass_rebinds_without_second_solver(root, monkeypatch, other_actor):
    import baloot_platform.research.ortools_information_set_compiler as observed_module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=raw["projectActions"][:2 if other_actor else 1],
        playActions=[{"actor": "north", "card": "S7", "ekka": False}] if other_actor else [])
    prefix = ExactTracePrefixV1.build(raw)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    deck = tuple(Card(index) for index in range(32))
    hands = {seat: tuple(deck[index] for index in layout.positions_for(seat, final=True)) for seat in Seat}
    state = {"contract": "sun", "trumpSuit": None, "projects": [], "projectWinningTeam": None}
    compiler = observed_module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    monkeypatch.setattr(observed_module, "attach_project_lifecycle_fragment_v1",
        lambda *args, **kwargs: pytest.fail("optional pass rebind must not solve projects twice"))
    rebound = compiler._rebind_project_actions(prefix, hands=hands, state=state,
        deadline=time.monotonic() + 10)
    assert rebound == tuple(prefix.to_dict()["projectActions"])


def test_optional_pass_other_actor_hidden_world_uses_legal_project_selector(root, monkeypatch):
    import baloot_platform.research.ortools_information_set_compiler as observed_module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=[{"actor": "north", "action": "pass"}],
        playActions=[{"actor": "north", "card": "S7", "ekka": False}])
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=30)
    assert exact.decision_kind == "project" and exact.next_actor == "west"
    monkeypatch.setattr(observed_module, "attach_project_lifecycle_fragment_v1",
        lambda *args, **kwargs: pytest.fail("optional pass must not constrain hidden ownership"))
    worlds = observed_module.OrToolsActorInformationSetCompilerV1(
        rules, native_cp_sat=True).enumerate_graph_world_inputs(
            exact, prefix, {"kind": "project", "selector": {"owner": "west"}},
            maximum_worlds=1, maximum_seconds=30, bounded_world_scope=True)
    assert len(worlds) == 1
    assert worlds[0][0].decision_kind == "project"


@pytest.mark.real_clock
def test_optional_all_pass_play_world_keeps_native_project_verification(root, monkeypatch):
    import baloot_platform.research.ortools_information_set_compiler as observed_module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["projectActions"].append({"actor": "east", "action": "pass"})
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    for _ in range(2):
        current = normalizer.normalize_play_or_project(prefix, maximum_seconds=15)
        prefix = normalizer.append_normalized_action(prefix, current, current.legal_actions[0]["code"])
    exact = normalizer.normalize_play_or_project(prefix, maximum_seconds=15)
    assert exact.decision_kind == "play" and len(prefix.play_actions) == 5
    original = observed_module.attach_project_lifecycle_fragment_v1
    project_models = []

    def verified_project_fragment(*args, **kwargs):
        project_models.append(kwargs.get("name"))
        return original(*args, **kwargs)

    monkeypatch.setattr(observed_module, "attach_project_lifecycle_fragment_v1", verified_project_fragment)
    worlds, allocation = observed_module.OrToolsActorInformationSetCompilerV1(
        rules, native_cp_sat=True)._enumerate_worlds(exact, maximum_worlds=1,
            maximum_attempts=100, deadline=time.monotonic() + 30,
            cancel_check=None, bounded_world_scope=True)
    assert len(worlds) == 1 and allocation["acceptedWorlds"] == 1
    assert project_models and set(project_models) == {"information_project"}


@pytest.mark.parametrize("through_post", [False, True, "round"])
def test_partial_project_boundary_preserves_declaration_then_generates_play(root, monkeypatch, through_post):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["projectActions"] = raw["projectActions"][:2]
    raw["playActions"] = raw["playActions"][:2]
    if through_post:
        raw["projectActions"] = []
        raw["playActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    submitted = ExactTracePrefixV1.build({**raw, "projectActions": [*raw["projectActions"],
        {"actor": "north" if through_post else "south", "action": "pass"}]})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} for kind in SCENARIO_DECISION_KINDS}})
    original = executor_module.compile_observed_decision_v1
    calls = []
    retained_plays = []
    def stop_after_pair(*args, **kwargs):
        calls.append(args[3]["kind"])
        played_calls = [kind for kind in calls if kind != "bid"]
        if len(played_calls) > 2:
            raise InformationSetCompilationStopped("cancelled", stage="action_analysis")
        assert (args[4].selected_action is not None) == (args[3]["kind"] == "bid" or len(played_calls) == 1)
        return original(*args, **kwargs)
    def stop_after_generated_play(row):
        if row.get("kind") == "quality_decision_checkpoint_v1" and row.get("stage") == "play":
            retained_plays.append(row)
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", stop_after_pair)
    parameters = dict(
        recorded_trace=submitted, continue_recorded_prefix=True,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=60,
        maximum_decision_seconds=30, maximum_evidence_bytes=10_000_000)
    if through_post:
        run = executor_module.resolve_quality_post_auction_continuation_v1
        if through_post == "round":
            initial = prefix.to_dict()
            initial.pop("sha256")
            initial["auctionActions"] = []
            prefix = ExactTracePrefixV1.build(initial)
            run = executor_module.resolve_quality_round_continuation_v1
        envelope = run(rules, prefix,
            {"kind": "escalation", "selector": {}}, policy, recorded_events=[],
            opportunity_order=("west", "east", "south", "north"), maximum_escalation_decisions=20,
            execution_budget=ScenarioQualityExecutionBudgetV1.build({
                "schemaVersion": 1,
                "maximumWallSeconds": parameters["maximum_seconds"],
                "maximumEvidenceBytes": parameters["maximum_evidence_bytes"],
                "maximumEscalationDecisions": 20,
                "earlyPlayPolicyId": "observer_safe_trick_control_v1",
                "exactPlayFromTrick": 7,
            }),
            checkpoint_callback=stop_after_generated_play,
            cancel_check=lambda: bool(retained_plays),
            **parameters)
        assert envelope["submittedPrefix"] == submitted.to_dict()
        assert envelope["operation"] == "audit_then_generate"
        if through_post == "round":
            from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
            validate_native_round_quality_evidence_v1(envelope,
                source_prefix=prefix, rules_hash=rules.sha256, policy=policy,
                decision_point={"kind": "escalation", "selector": {}})
            envelope = envelope["postAuction"]
        assert envelope["escalation"]["operation"] == "audit_then_generate"
        result = envelope["projectPlay"]
    else:
        result = resolve_quality_play_continuation_v1(rules, prefix, policy, **parameters)
    assert result["status"] == "incomplete", result["partialReceipt"]
    assert calls == (
        (["bid"] if through_post == "round" else [])
        + (["project"] if through_post else ["project", "play", "project"])
    )
    assert result["decisions"][0]["nextPrefixHash"] == submitted.sha256
    assert result["decisions"][0]["selectionReason"] == "recorded_action"
    assert result["decisions"][1]["selectionReason"] == (
        "native_observer_safe_policy_value"
        if through_post
        else "native_actor_information_set_best_value"
    )
    assert result["finalPrefix"]["projectActions"] == submitted.to_dict()["projectActions"]
    assert result["finalPrefix"]["playActions"][:len(raw["playActions"])] == raw["playActions"]
    assert len(result["finalPrefix"]["playActions"]) == len(raw["playActions"]) + 1


@pytest.mark.real_clock
def test_complete_state_policy_covers_bids_projects_and_all_plays(root, record_property):
    from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=[], playActions=[])
    submitted = ExactTracePrefixV1.build(raw)
    initial = ExactTracePrefixV1.build({**raw, "auctionActions": []})
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "visibility": "complete_state", "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} for kind in SCENARIO_DECISION_KINDS},
    })
    decision = {"kind": "play", "selector": {}}
    limits = dict(opportunity_order=("west", "east", "south", "north"),
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=180, maximum_decision_seconds=30, maximum_escalation_decisions=20,
        maximum_evidence_bytes=30_000_000)
    result = executor_module.resolve_quality_round_continuation_v1(
        rules, initial, decision, policy, recorded_trace=submitted, recorded_events=[],
        continue_recorded_prefix=True, **limits)
    assert result["status"] == "complete", result["partialReceipt"]
    timing = result["executionReceipt"]
    assert timing["auction"]["elapsedSeconds"] >= 0
    assert timing["postAuction"]["projectPlay"]["elapsedSeconds"] >= 0
    record_property("qualityPhaseSeconds", json.dumps({
        "auction": timing["auction"]["elapsedSeconds"],
        "postAuction": timing["postAuction"]["elapsedSeconds"],
        "projectPlay": timing["postAuction"]["projectPlay"]["elapsedSeconds"],
    }, sort_keys=True))
    decision_timings = timing["postAuction"]["projectPlay"]["decisionTimings"]
    assert len(decision_timings) >= 32
    record_property("qualityDecisionSeconds", json.dumps({
        kind: {field: round(sum(row.get(field, 0.0) for row in decision_timings
                               if row["decisionKind"] == kind), 3)
               for field in ("compilationSeconds", "proofSeconds", "compilationAndProofSeconds",
                             "selectedActionNormalizationSeconds")}
        for kind in {row["decisionKind"] for row in decision_timings}
    }, sort_keys=True))
    validate_native_round_quality_evidence_v1(result, source_prefix=initial,
        rules_hash=rules.sha256, policy=policy, decision_point=decision)
    play = result["postAuction"]["projectPlay"]
    assert len(result["finalPrefix"]["playActions"]) == 32
    assert {row["decisionKind"] for row in play["decisions"]} == {"project", "play"}
    assert len([row for row in play["decisions"] if row["decisionKind"] == "play"]) == 32
    assert len(result["auction"]["decisions"]) == len(submitted.auction_actions)
    for stage in (result["auction"], play):
        for row in stage["decisions"]:
            artifact = stage["artifacts"][row["proofArtifactHash"]]
            assert artifact["request"]["visibility"] == "complete_state"
            assert artifact["proof"]["completionStatus"] == "bounded_complete"
            assert artifact["proofModel"]["actionQuality"]["actualActionOptimal"] is True
    audit = executor_module.resolve_quality_round_continuation_v1(
        rules, initial, decision, policy, recorded_trace=ExactTracePrefixV1.build(result["finalPrefix"]),
        recorded_events=result["postAuction"]["escalation"]["events"], **limits)
    assert audit["status"] == "complete", audit["partialReceipt"]
    assert audit["finalPrefix"] == result["finalPrefix"]


@pytest.mark.parametrize("through_entry,auction_only,recorded_decline", [
    (False, False, False), (True, False, False), (False, True, False), (True, True, False), (True, True, True)])
def test_complete_round_preserves_partial_prefix_and_normalizes_candidate(root, through_entry, auction_only, recorded_decline, tmp_path):
    from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=[], playActions=[])
    if recorded_decline:
        raw["auctionActions"] = [{"action": action} for action in ("hokum", "pass", "pass", "pass")]
    submitted = ExactTracePrefixV1.build({**raw, "projectActions": [{"actor": "north", "action": "pass"}]})
    if auction_only:
        submitted = ExactTracePrefixV1.build(raw)
    initial = ExactTracePrefixV1.build({**raw, "auctionActions": []})
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "tricks" if through_entry and kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} for kind in SCENARIO_DECISION_KINDS}})
    events = [{"actor": "west", "actionCode": "decline"}] if recorded_decline else []
    selected_actor = "east" if recorded_decline else "west"
    decision = {"kind": "escalation" if auction_only else "play",
        "selector": {"actingSeat": selected_actor} if auction_only else {}}
    if through_entry:
        from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2
        profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
        budget = copy.deepcopy(profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"])
        budget.update(maximumNodes=1000, maximumDepth=8, maximumHiddenWorlds=2 if recorded_decline else 1,
            maximumWallSeconds=60, maximumCpuSeconds=60)
        profile["scenarioResearchV4"]["proofBudgets"]["profiles"] = {"test": budget}
        backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity={
            "bundleRevisionId": "CBR-PARTIAL-ROUND-TEST", "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64})
        entry = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-PARTIAL-ROUND-TEST", bundle_hash="a" * 64,
            quality_policy=policy, quality_execution=backend.authority.quality_execution,
            quality_proof_profiles={"test": budget})
        scenario = _scenario(
            constraints=_leaf("PRED-CONTRACT", "auction.contract", {"contract": "hokum" if recorded_decline else "sun"}),
            outcome=_leaf("PRED-OUTCOME", "outcome.team_round", {"team": "north_south", "status": "won"}, "report"),
            origin={"kind": "exact_trace_prefix", "exactTracePrefix": submitted.to_dict(),
                    "recordedEscalationOpportunities": events},
            decision_point=decision)
        proof_raw = _bounded_proof().to_dict()
        proof_raw.pop("sha256")
        proof_raw["successThreshold"] = 0
        checkpoints = []
        attempt = entry.execute(scenario, ScenarioProofRequestV1.build(proof_raw), maximum_seconds=180,
            maximum_proof_nodes=1000, maximum_proof_depth=8,
            checkpoint_callback=lambda row: checkpoints.append(row["nextPrefix"])
                if row.get("kind") == "quality_decision_checkpoint_v1" and "nextPrefix" in row else None)
        failure = attempt.execution_receipt.get("incompleteCompilation") or {}
        quality = failure.get("qualityEvidence") or {}
        post = quality.get("postAuction") or {}
        play = post.get("projectPlay") or {}
        play_execution = ((failure.get("executionReceipt") or {}).get("postAuction") or {}).get("projectPlay") or {}
        timings = play_execution.get("decisionTimings") or []
        cause = play.get("partialReceipt") or {}
        cause_chain = []
        for _ in range(8):
            if not isinstance(cause, dict):
                break
            cause_chain.append((cause.get("stage"), cause.get("exhaustionReason") or cause.get("reason")))
            cause = (cause.get("evidence") or {}).get("causeReceipt")
        last_prefix_path = tmp_path / "last_quality_prefix.json"
        if attempt.status != "qualified" and checkpoints:
            last_prefix_path.write_text(json.dumps(checkpoints[-1], sort_keys=True), encoding="utf-8")
        assert attempt.status == "qualified", {
            "stageTimings": attempt.execution_receipt.get("stageTimings"),
            "qualityFailureStage": failure.get("qualityFailureStage"),
            "qualityFailureReason": failure.get("qualityFailureReason"),
            "playStatus": play.get("status"), "causeChain": cause_chain,
            "lastPrefixPath": str(last_prefix_path) if checkpoints else None,
            "decisions": len(play.get("decisions") or []),
            "lastDecisionTiming": timings[-1] if timings else None,
        }
        selected = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            submitted, decision_point=decision, maximum_seconds=30,
            **({"recorded_escalation_opportunities": events,
                "opportunity_order": backend.authority.quality_execution.escalation_opportunity_order}
               if recorded_decline else {}))
        assert attempt.candidate.trace["exactNormalization"]["exactStateId"] == selected.exact_state_id
        assert attempt.candidate.exact_state_id == selected.exact_state_id
        assert attempt.candidate.trace["exactTracePrefix"] == submitted.to_dict()
        assert attempt.candidate.trace["completeExactNormalization"]["exactStateId"] != selected.exact_state_id
        from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
        store = LocalContentAddressedArtifactStore(tmp_path / "exact-quality-artifacts")
        documents = attempt.semantic_artifact_documents()
        references = {name: store.put(json.dumps(document, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8"), media_type="application/json", name=f"{name}.json")
            for name, document in documents.items()}
        reopened = LocalContentAddressedArtifactStore(tmp_path / "exact-quality-artifacts")
        loaded = {name: json.loads(reopened.get(reference["uri"])) for name, reference in references.items()}
        assert loaded == documents
        def rebuild(values):
            return OrToolsResearchArtifactPreimagesV1.build(authority=attempt.research_authority,
                construction_model=values["constructionModel"], proof_model=values["proofModel"],
                proof_graph=values["proofGraph"], trace=values["trace"], proof_outcome=values["proofOutcome"])
        assert rebuild(loaded).to_dict() == documents
        executor_module.validate_trace_quality_evidence_v1(loaded["trace"], rules_hash=rules.sha256, rules=rules,
            required_policy=policy, required_execution=backend.authority.quality_execution,
            required_proof_profiles={"test": budget})
        changed = copy.deepcopy(loaded)
        changed["trace"]["exactTracePrefix"] = changed["trace"]["completeExactTracePrefix"]
        with pytest.raises(ValueError, match="trace"):
            rebuild(changed)
        result = attempt.candidate.trace["playingQualityEvidence"]["receipt"]
        assert result["qualityExecution"] == backend.authority.quality_execution.to_dict()
    else:
        result = executor_module.resolve_quality_round_continuation_v1(rules, initial, decision, policy,
            recorded_trace=submitted, recorded_events=[], continue_recorded_prefix=True,
            opportunity_order=("west", "east", "south", "north"), maximum_escalation_decisions=20,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=180,
            maximum_decision_seconds=30, maximum_evidence_bytes=30_000_000)
    assert result["status"] == "complete", result["partialReceipt"]
    assert len(result["finalPrefix"]["playActions"]) == 32
    assert result["finalPrefix"]["projectActions"][:len(submitted.project_actions)] == submitted.to_dict()["projectActions"]
    assert result["finalPrefix"]["auctionActions"] == submitted.to_dict()["auctionActions"]
    if through_entry:
        # The loaded artifact already received a cold full-round verification above.
        candidate = attempt.candidate
    else:
        validate_native_round_quality_evidence_v1(result, source_prefix=initial,
            rules_hash=rules.sha256, rules=rules, policy=policy, decision_point=decision)
        receipt = {key: value for key, value in result.items() if key != "executionReceipt"}
        trace = {"completeExactTracePrefix": result["finalPrefix"], "playingQualityEvidence": {
            "sourcePrefix": initial.to_dict(), "decisionPoint": decision, "receipt": receipt}}
        executor_module.validate_trace_quality_evidence_v1(trace, rules_hash=rules.sha256, rules=rules, required_policy=policy)
        executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-PARTIAL-ROUND-TEST", bundle_hash="a" * 64)
        candidate = executor.candidate_from_quality_trace(trace, maximum_seconds=30)
    assert candidate.origin == "exact_trace_prefix"
    assert candidate.trace["completeExactTracePrefix"] == result["finalPrefix"]
    assert candidate.trace["playingQualityEvidence"]["receipt"]["submittedPrefix"] == submitted.to_dict()


def test_native_quality_audits_recorded_interleaved_project_and_play(root, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["projectActions"] = raw["projectActions"][:2]
    raw["playActions"] = raw["playActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    recorded = _optional_project_prefix_v1()
    recorded["projectActions"].append({"actor": "east", "action": "pass"})
    played = {row["card"] for row in recorded["playActions"]}
    recorded["playActions"].extend(
        {"card": card.code, "ekka": False} for card in map(Card, range(32)) if card.code not in played
    )
    trace = ExactTracePrefixV1.build(recorded)
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS},
    })
    original = executor_module.compile_observed_decision_v1
    calls = []

    def stop_after_recorded_pair(*args, **kwargs):
        calls.append(args[3]["kind"])
        if len(calls) > 2:
            raise InformationSetCompilationStopped("cancelled", stage="action_analysis")
        assert args[4].selected_action is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", stop_after_recorded_pair)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, recorded_trace=trace,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=10_000_000,
    )
    assert calls == ["project", "play", "project"], result
    assert result["status"] == "incomplete"
    assert result["operation"] == "audit_recorded"
    assert result["recordedPrefixHash"] == trace.sha256
    assert [row["decisionKind"] for row in result["decisions"]] == ["project", "play"]
    _assert_quality_ownership_rejects_mutations(result, prefix, rules, policy)
    assert result["decisions"][1]["action"] == "SK"
    for row in result["decisions"]:
        assert row["selectionReason"] == "recorded_action"
        assert row["selectionTieBreak"] is None
        quality = result["artifacts"][row["proofArtifactHash"]]["proofModel"]["actionQuality"]
        assert quality["actualAction"] == row["action"]
    assert result["finalPrefix"]["projectActions"] == recorded["projectActions"][:3]
    assert result["finalPrefix"]["playActions"] == recorded["playActions"][:3]
    assert prefix.to_dict()["playActions"] == raw["playActions"]
    assert trace.to_dict()["playActions"] == recorded["playActions"]


@pytest.mark.real_clock
@pytest.mark.parametrize("outer_nodes,outer_depth", [(1000, 8), (300, 1)])
@pytest.mark.parametrize("clock_start", [None, 255.99999999999997])
def test_native_quality_applies_each_decisions_frozen_budget(root, monkeypatch, outer_nodes, outer_depth, clock_start):
    from types import SimpleNamespace
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    if clock_start is not None:
        assert clock_start + 15 - clock_start > 15
        monkeypatch.setattr(executor_module, "time", SimpleNamespace(
            **{**vars(executor_module.time), "monotonic": lambda: clock_start}))

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    profiles = {
        "project": _quality_budget_profile(root, maximumNodes=500, maximumDepth=2, maximumWallSeconds=15),
        "play": _quality_budget_profile(root, maximumNodes=600, maximumDepth=3, maximumWallSeconds=5),
    }
    preserved = copy.deepcopy(profiles)
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "project" if kind == "project" else "play",
            "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in SCENARIO_DECISION_KINDS},
    })
    original = executor_module.compile_observed_decision_v1
    seen = []

    def inspect_limits(*args, **kwargs):
        kind = args[3]["kind"]
        seen.append(kind)
        expected = preserved[kind]
        assert callable(kwargs["cancel_check"])
        assert kwargs["maximum_nodes"] == min(outer_nodes, expected["maximumNodes"])
        assert kwargs["maximum_depth"] == min(outer_depth, expected["maximumDepth"])
        assert kwargs["maximum_worlds"] == expected["maximumHiddenWorlds"]
        assert 0 < kwargs["maximum_seconds"] <= expected["maximumWallSeconds"]
        if kind == "play":
            raise InformationSetCompilationStopped("cancelled", stage="action_analysis")
        profiles["play"]["maximumNodes"] = 999
        profiles["play"]["maximumWallSeconds"] = 999
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", inspect_limits)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, proof_budget_profiles=profiles,
        maximum_nodes=outer_nodes, maximum_depth=outer_depth, maximum_worlds=64,
        maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=10_000_000,
    )
    assert seen == ["project", "play"], result
    assert result["status"] == "incomplete"
    assert result["proofBudgetProfiles"] == preserved
    assert result["decisions"][0]["proofBudgetHash"] == ScenarioProofBudgetV1.build("project", preserved["project"]).sha256


@pytest.mark.parametrize("mutation,reason", [
    ("context", "source context"),
    ("prefix", "source actions"),
    ("short", "complete 32-play"),
    ("both", "not both"),
    ("project_semantics", "project action semantics"),
])
def test_native_quality_recorded_trace_rejects_changed_inputs_before_solving(root, rules, monkeypatch, mutation, reason):
    prefix, request = _quality_continuation_input(root, "bounded")
    document = prefix.to_dict()
    document.pop("sha256")
    document["playActions"].extend({"card": card, "ekka": False} for card in ("SA", "C9", "DA", "D8", "S7"))
    if mutation == "context":
        document["matchScore"] = [1, 0]
    elif mutation == "prefix":
        document["playActions"][0]["card"] = "H7"
    elif mutation == "short":
        document["playActions"].pop()
    elif mutation == "project_semantics":
        document["projectActions"] = []

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid recording reached normalization")

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", forbidden)
    with pytest.raises(ValueError, match=reason):
        resolve_quality_play_continuation_v1(
            rules, prefix, request, recorded_trace=ExactTracePrefixV1.build(document),
            **({"recorded_actions": []} if mutation == "both" else {}),
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
            maximum_seconds=60, maximum_decision_seconds=30, maximum_evidence_bytes=10_000_000,
        )


@pytest.mark.parametrize("phase", ["scored", "playing"])
def test_native_quality_recorded_trace_validates_final_assertions(root, rules, phase):
    prefix, request = _quality_continuation_input(root, "bounded")
    document = prefix.to_dict()
    document.pop("sha256")
    document["playActions"].extend({"card": card, "ekka": False} for card in ("SA", "C9", "DA", "D8"))
    prefix = ExactTracePrefixV1.build(document)
    document["playActions"].append({"card": "S7", "ekka": False})
    document["stateAssertions"] = {"phase": phase}
    trace = ExactTracePrefixV1.build(document)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, request, recorded_trace=trace,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=15, maximum_evidence_bytes=10_000_000,
    )
    assert result["recordedPrefixHash"] == trace.sha256
    if phase == "scored":
        assert result["status"] == "complete", result
        assert result["finalPrefix"] == trace.to_dict()
    else:
        assert result["status"] == "model_error", result
        assert "assertion mismatch at phase" in result["partialReceipt"]["reason"]
        assert result["finalPrefix"] == prefix.to_dict()
        assert result["decisions"] == []


@pytest.mark.parametrize("declarations,project_unit", [(2, "plies"), (2, "tricks"), (3, "plies")])
def test_native_quality_selects_project_then_preserves_receipt_on_cancellation(root, monkeypatch, declarations, project_unit):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["projectActions"] = raw["projectActions"][:declarations]
    raw["playActions"] = raw["playActions"][:declarations]
    prefix = ExactTracePrefixV1.build(raw)
    actor = "south" if declarations == 2 else "east"
    team = "north_south" if declarations == 2 else "west_east"
    project_horizon = 3 if declarations == 2 and project_unit == "plies" else 1
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "plies" if kind == "play" else project_unit,
                        "value": 2 if kind == "play" else project_horizon},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in ("bid", "escalation", "project", "play")},
    })
    calls = []
    original = executor_module.compile_observed_decision_v1

    def stop_after_project(*args, **kwargs):
        calls.append(args[3]["kind"])
        assert args[4].horizon["value"] == (2 if args[3]["kind"] == "play" else project_horizon)
        assert args[4].horizon["unit"] == ("plies" if args[3]["kind"] == "play" else project_unit)
        assert args[4].target_team == team
        if len(calls) > 1:
            raise InformationSetCompilationStopped("cancelled", stage="action_analysis")
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", stop_after_project)
    checkpoints = []
    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, maximum_nodes=1000, maximum_depth=8,
        maximum_worlds=1, maximum_seconds=60, maximum_decision_seconds=30,
        maximum_evidence_bytes=10_000_000,
        checkpoint_callback=checkpoints.append,
    )
    assert calls == ["project", "play"], result
    assert result["status"] == "incomplete"
    assert result["scope"] == "project_play_suffix"
    assert len(result["decisions"]) == 1
    row = result["decisions"][0]
    assert len(checkpoints) == 2
    assert checkpoints[1]["stage"] == "project"
    assert checkpoints[1]["decision"] == row
    assert checkpoints[1]["nextPrefix"] == result["finalPrefix"]
    assert row["decisionKind"] == "project"
    assert row["actor"] == actor
    assert row["targetTeam"] == team
    assert row["actualActionOptimal"] is True
    assert result["artifacts"][row["proofArtifactHash"]]["proof"]["completionStatus"] == "bounded_complete"
    assert len(result["finalPrefix"]["projectActions"]) == declarations + 1
    assert result["finalPrefix"]["playActions"] == prefix.to_dict()["playActions"]
    assert len(prefix.project_actions) == declarations
    limits = dict(maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
                  maximum_seconds=60, maximum_decision_seconds=30,
                  maximum_evidence_bytes=10_000_000)
    with pytest.raises(ValueError, match="per-decision policy"):
        resolve_quality_play_continuation_v1(
            rules, prefix, policy.request_for("play", "east"), **limits,
        )
    with pytest.raises(ValueError, match="recorded play-only audit"):
        resolve_quality_play_continuation_v1(rules, prefix, policy, recorded_actions=[], **limits)
    assert calls == ["project", "play"]


@pytest.mark.parametrize("root_kind", ["project", "play"])
def test_exact_quality_preserves_next_project_decision(root, monkeypatch, root_kind):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["playActions"] = raw["playActions"][:2]
    if root_kind == "project":
        raw["projectActions"] = raw["projectActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "visibility": "complete_state", "mode": "exact",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test"}
                      for kind in SCENARIO_DECISION_KINDS},
    })
    visited = []
    original = OrToolsExactStateNormalizerV1.normalize_play_or_project

    def inspect_successor(self, *args, **kwargs):
        normalized = original(self, *args, **kwargs)
        visited.append((normalized.decision_kind, normalized.next_actor))
        return normalized

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", inspect_successor)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, maximum_nodes=1000, maximum_depth=40, maximum_worlds=1,
        maximum_seconds=30, maximum_decision_seconds=20, maximum_evidence_bytes=10_000_000,
        cancel_check=lambda: ("project", "east") in visited,
    )
    assert visited[0] == (root_kind, "south")
    assert ("project", "east") in visited
    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == "cancelled"
    assert result["decisions"] == []
    assert result["finalPrefix"] == prefix.to_dict()


@pytest.mark.parametrize("unit,value", [("plies", 2), ("tricks", 1)])
@pytest.mark.parametrize("worlds", [1, 2])
def test_native_quality_play_looks_through_next_declaration(root, monkeypatch, unit, value, worlds):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["playActions"] = raw["playActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": unit if kind == "play" else "plies", "value": value if kind == "play" else 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in ("bid", "escalation", "project", "play")},
    })
    original = executor_module.compile_observed_decision_v1
    calls = []

    def stop_after_play(*args, **kwargs):
        calls.append(args[3]["kind"])
        if len(calls) > 1:
            raise InformationSetCompilationStopped("cancelled", stage="action_analysis")
        assert args[4].target_team == "north_south"
        assert args[4].horizon == {"unit": unit, "value": value}
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", stop_after_play)
    result = resolve_quality_play_continuation_v1(
        rules, prefix, policy, maximum_nodes=1000, maximum_depth=8,
        maximum_worlds=worlds, maximum_seconds=60, maximum_decision_seconds=30,
        maximum_evidence_bytes=10_000_000,
    )
    assert calls == ["play", "project"], result
    assert result["status"] == "incomplete"
    assert len(result["decisions"]) == 1
    row = result["decisions"][0]
    assert row["decisionKind"] == "play"
    assert row["actor"] == "south"
    assert row["actualActionOptimal"] is True
    artifact = result["artifacts"][row["proofArtifactHash"]]
    assert artifact["proof"]["completionStatus"] == "bounded_complete"
    assert artifact["proofModel"]["rootActionCountScope"] == "declared_worlds_only"
    assert artifact["graph"]["hiddenWorlds"]["count"] == worlds
    assert artifact["graph"]["hiddenWorlds"]["complete"] is False
    assert len(result["finalPrefix"]["playActions"]) == 3
    assert result["finalPrefix"]["projectActions"] == prefix.to_dict()["projectActions"]
    assert len(prefix.play_actions) == 2


@pytest.mark.real_clock
@pytest.mark.parametrize("complete_round", [False, True])
def test_native_quality_generates_all_first_trick_decisions(root, monkeypatch, complete_round):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    raw["projectActions"] = []
    raw["playActions"] = []
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({
            "schemaVersion": 1, "mode": "budgeted",
            "decisions": {kind: {
                "utility": "awarded_point_margin", "budgetProfile": "test",
                "horizon": {"unit": "tricks" if kind == "play" else "plies",
                            "value": 2 if kind == "project" else 1},
                "boundaryEvaluator": "current_raw_point_margin_v1",
        } for kind in ("bid", "escalation", "project", "play")},
    })
    original = executor_module.compile_observed_decision_v1
    calls = []
    checkpoints = []

    def record_quality_checkpoint(row):
        if row.get("kind") == "quality_proof_checkpoint_v1" and row.get("stage") in {"project", "play"}:
            checkpoints.append(row)

    def stop_after_first_trick(*args, **kwargs):
        calls.append(args[3]["kind"])
        assert args[4].horizon["value"] == (2 if args[3]["kind"] == "project" else 1)
        return original(*args, **kwargs)

    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", stop_after_first_trick)
    limits = dict(maximum_nodes=1000, maximum_depth=8, maximum_worlds=1,
                  maximum_seconds=300 if complete_round else 180, maximum_decision_seconds=30,
                  maximum_evidence_bytes=30_000_000)
    def run(recorded_trace=None):
        execution_budget = ScenarioQualityExecutionBudgetV1.build({
            "schemaVersion": 1, "maximumWallSeconds": limits["maximum_seconds"],
            "maximumEvidenceBytes": limits["maximum_evidence_bytes"],
            "maximumEscalationDecisions": 20,
            "earlyPlayPolicyId": "observer_safe_trick_control_v1",
            "exactPlayFromTrick": 7,
        }) if complete_round else None
        connected = executor_module.resolve_quality_post_auction_continuation_v1(rules, prefix,
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
            opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
            recorded_trace=recorded_trace, recorded_events=[] if recorded_trace is not None else None,
            checkpoint_callback=record_quality_checkpoint,
            cancel_check=(lambda: not complete_round and bool(checkpoints)),
            execution_budget=execution_budget, **limits)
        assert connected["scope"] == "post_auction_only"
        assert connected["escalation"]["status"] == "complete"
        assert connected["executionReceipt"]["evidenceBytes"] <= limits["maximum_evidence_bytes"]
        assert connected["sha256"] == canonical_sha256({
            key: value for key, value in connected.items() if key not in {"sha256", "executionReceipt"}})
        assert connected["projectPlay"]["limits"]["maximumSeconds"] == limits["maximum_seconds"]
        assert connected["projectPlay"]["limits"]["maximumEvidenceBytes"] < limits["maximum_evidence_bytes"]
        return connected["projectPlay"]
    result = run()
    if complete_round:
        assert 0 < calls.count("project") <= 4
        assert 0 < calls.count("play") <= 8, str(result.get("partialReceipt"))[:1000]
    else:
        assert calls == ["project"]
    assert result["status"] == ("complete" if complete_round else "incomplete"), result.get("partialReceipt")
    assert len(result["decisions"]) == (36 if complete_round else 0)
    for index, row in enumerate(result["decisions"][:8] if complete_round else result["decisions"]):
        actor = (Seat.NORTH, Seat.WEST, Seat.SOUTH, Seat.EAST)[index // 2]
        assert row["actor"] == actor.name.lower()
        assert row["targetTeam"] == actor.team.name.lower()
        assert row["decisionKind"] == ("project" if index % 2 == 0 else "play")
        assert row["actualActionOptimal"] is True
        assert result["artifacts"][row["proofArtifactHash"]]["proof"]["completionStatus"] == "bounded_complete"
        if complete_round and row["decisionKind"] == "play":
            assert row["selectionReason"] in {
                "native_actor_information_set_best_value",
                "native_observer_safe_policy_value",
            }
    assert len(result["finalPrefix"]["projectActions"]) == (4 if complete_round else 0)
    assert len(result["finalPrefix"]["playActions"]) == (32 if complete_round else 0)
    assert len(prefix.project_actions) == len(prefix.play_actions) == 0
    if complete_round:
        trace = ExactTracePrefixV1.build(result["finalPrefix"])
        initial_raw = prefix.to_dict()
        initial_raw.pop("sha256")
        initial_raw["auctionActions"] = []
        initial = ExactTracePrefixV1.build(initial_raw)
        from baloot_platform.research.scenario_backend import OrToolsScenarioResearchBackendV2

        profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))
        proof_profile = copy.deepcopy(profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["late_game_exact_v1"])
        proof_profile.update(maximumNodes=1000, maximumDepth=8, maximumHiddenWorlds=1,
                             maximumWallSeconds=30, maximumCpuSeconds=30)
        profile["scenarioResearchV4"]["proofBudgets"]["profiles"] = {"test": proof_profile}
        profile["scenarioResearchV4"]["qualityExecution"].update(
            earlyPlayPolicyId="observer_safe_trick_control_v1",
            exactPlayFromTrick=7, pruneOwnProjectPassWorlds=False,
        )
        backend = OrToolsScenarioResearchBackendV2(challenge_profile=profile, bundle_identity={
            "bundleRevisionId": "CBR-QUALITY-ROUND-TEST", "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64})
        original_trace = {"completeExactTracePrefix": trace.to_dict()}
        attached = backend.audit_quality_trace(
            rules, original_trace, initial, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
            recorded_events=[], maximum_seconds=limits["maximum_seconds"])
        full_audit = attached["quality"]
        assert original_trace == {"completeExactTracePrefix": trace.to_dict()}
        assert attached["trace"]["completeExactTracePrefix"] == trace.to_dict()
        assert attached["trace"]["playingQualityEvidence"]["receipt"]["finalPrefix"] == trace.to_dict()
        assert full_audit["status"] == "complete", full_audit["partialReceipt"]
        quality_executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-QUALITY-ROUND-TEST", bundle_hash="a" * 64,
            quality_policy=policy, quality_execution=backend.authority.quality_execution,
            quality_proof_profiles={"test": proof_profile})
        candidate = quality_executor.candidate_from_quality_trace(attached["trace"], maximum_seconds=30)
        assert candidate.origin == "exact_trace_prefix"
        assert candidate.trace["completeExactTracePrefix"] == trace.to_dict()
        assert candidate.trace["playingQualityEvidence"] == attached["trace"]["playingQualityEvidence"]
        assert candidate.trace["playingQualityEvidence"]["receipt"]["operation"] == "audit_recorded"
        from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1
        receipt = copy.deepcopy(full_audit)
        receipt["sha256"] = receipt.pop("sourceQualityHash")
        receipt.pop("backendIdentity")
        receipt.pop("configurationAuthority")
        terminal = OrToolsExactStateNormalizerV1(
            rules,
            native_cp_sat=True,
        ).normalize(
            trace,
            decision_point={"kind": "play", "selector": {}},
            maximum_seconds=15,
        )
        def retained_audit(*args, **kwargs):
            assert args[1] == initial
            assert kwargs["recorded_trace"] == trace
            assert kwargs["recorded_events"] == []
            assert receipt["finalPrefix"] == trace.to_dict()
            kwargs["final_normalized_callback"](terminal)
            return receipt
        with monkeypatch.context() as audit_patch:
            audit_patch.setattr(executor_module, "resolve_quality_round_continuation_v1", retained_audit)
            attempt = quality_executor.audit_quality_candidate(candidate, recorded_trace=trace,
                recorded_events=[], maximum_seconds=30,
                decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
        assert attempt.status == "feasible"
        assert attempt.candidate.origin == "exact_trace_prefix"
        assert attempt.candidate.trace["completeExactTracePrefix"] == trace.to_dict()
        with pytest.raises(ValueError, match="explicit opportunity history"):
            quality_executor.audit_quality_candidate(candidate, recorded_trace=trace,
                recorded_events=None, maximum_seconds=30, decision_point={"kind": "escalation"})
        ownership = dict(source_prefix=initial, rules_hash=rules.sha256, rules=rules, policy=policy,
            decision_point={"kind": "escalation", "selector": {"actingSeat": "west"}})
        validate_native_round_quality_evidence_v1(receipt, **ownership)
        changed = copy.deepcopy(receipt)
        changed["postAuction"] = None
        changed["finalPrefix"] = changed["auction"]["finalPrefix"]
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items()
                                             if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="round quality false completion"):
            validate_native_round_quality_evidence_v1(changed, **ownership)
        assert full_audit["scope"] == "all_decisions"
        assert full_audit["qualityExecution"] == backend.authority.quality_execution.to_dict()
        assert full_audit["auction"]["proofBudgetProfiles"] == {"test": proof_profile}
        assert full_audit["sourcePrefixHash"] == initial.sha256
        assert full_audit["finalPrefix"] == trace.to_dict()
        assert len(full_audit["auction"]["decisions"]) == len(trace.auction_actions)
        assert full_audit["executionReceipt"]["evidenceBytes"] <= limits["maximum_evidence_bytes"]
        audited = full_audit["postAuction"]["projectPlay"]
        assert audited["proofBudgetProfiles"] == {"test": proof_profile}
        assert audited["status"] == "complete", audited.get("partialReceipt")
        assert audited["finalPrefix"] == result["finalPrefix"]
        assert audited["finalExactStateId"] == result["finalExactStateId"]
        assert len(audited["decisions"]) == 36
        source_hash = prefix.sha256
        for generated, recorded in zip(result["decisions"], audited["decisions"], strict=True):
            assert recorded["sourcePrefixHash"] == generated["sourcePrefixHash"] == source_hash
            assert recorded["nextPrefixHash"] == generated["nextPrefixHash"]
            assert recorded["action"] == generated["action"]
            assert recorded["decisionKind"] == generated["decisionKind"]
            assert recorded["actor"] == generated["actor"]
            assert recorded["actualActionOptimal"] is True
            assert recorded["regret"] == 0
            assert recorded["selectionReason"] == "recorded_action"
            source_hash = recorded["nextPrefixHash"]
        assert source_hash == trace.sha256


@pytest.mark.parametrize("mode,visibility", [
    ("exact", "actor_information_set"),
    ("exact", "complete_state"),
    ("bounded", "complete_state"),
])
def test_observer_safe_policy_cannot_supply_exact_or_complete_state_proof(rules, mode, visibility):
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": mode, "visibility": visibility,
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "successThreshold": 0, "targetTeam": "north_south",
        "utility": "awarded_point_margin", "budgetProfile": "test",
        **({"horizon": {"unit": "tricks", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} if mode == "bounded" else {}),
    })
    with pytest.raises(ValueError, match="cannot supply exact or complete-state proof"):
        executor_module._observer_safe_play_policy_artifact_v1(
            rules, None, request, actual_action=None, maximum_seconds=1, cancel_check=None)


def test_observer_safe_policy_ignores_hidden_hands_and_spends_ace_only_to_take_control(rules):
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "successThreshold": 0, "targetTeam": "north_south",
        "utility": "awarded_point_margin", "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    base_state = {
        "contract": "sun", "trumpSuit": None,
        "currentTrick": {"index": 0, "leader": "west", "plays": [
            {"actor": "west", "card": "H10"},
        ]},
        "hands": {"south": ["H7", "HA"], "west": ["S7"], "east": ["D7"], "north": ["C7"]},
    }
    legal = ({"code": "H7"}, {"code": "HA"})
    first = SimpleNamespace(decision_kind="play", next_actor="south", state=base_state, legal_actions=legal)
    changed = copy.deepcopy(base_state)
    changed["hands"]["west"], changed["hands"]["east"] = changed["hands"]["east"], changed["hands"]["west"]
    second = SimpleNamespace(decision_kind="play", next_actor="south", state=changed, legal_actions=legal)

    first_action, first_artifact = executor_module._observer_safe_play_policy_artifact_v1(
        rules, first, request, actual_action=None, maximum_seconds=1, cancel_check=None)
    second_action, second_artifact = executor_module._observer_safe_play_policy_artifact_v1(
        rules, second, request, actual_action=None, maximum_seconds=1, cancel_check=None)

    assert first_action == second_action == "HA"
    assert first_artifact["graph"] == second_artifact["graph"]
    assert first_artifact["allocationModel"] == second_artifact["allocationModel"]
    assert first_artifact["proofModel"]["actionQuality"]["scope"] == "observer_safe_policy_conditioned"


def test_complete_construction_prefix_allows_mandatory_project_declaration_extension(
    rules, complete_candidate,
):
    recorded = executor_module._construction_complete_prefix_v1(complete_candidate)
    source = ExactTracePrefixV1.build({
        "schemaVersion": 1,
        "deckOrder": list(recorded.deck_order),
        "dealer": recorded.dealer,
        "roundNumber": recorded.round_number,
        "matchScore": list(recorded.match_score),
        "auctionActions": [],
        "escalationActions": [],
        "projectActions": [],
        "projectDeclarations": [],
        "playActions": [],
    })
    assert recorded.project_actions_present
    assert executor_module._validate_quality_recorded_trace(
        source, recorded, allow_auction_extension=True,
        allow_escalation_extension=True,
    ) == recorded


def test_recorded_quality_trace_identifies_changed_source_field(rules, complete_candidate):
    recorded = executor_module._construction_complete_prefix_v1(complete_candidate)
    source_data = recorded.to_dict()
    source_data.pop("sha256")
    source_data["roundNumber"] = recorded.round_number + 1
    source = ExactTracePrefixV1.build(source_data)
    with pytest.raises(ValueError, match="source context: roundNumber"):
        executor_module._validate_quality_recorded_trace(source, recorded)


def test_nonoptimal_recorded_play_key_reuses_only_the_same_decision_prefix(
    complete_candidate,
):
    recorded = executor_module._construction_complete_prefix_v1(complete_candidate)
    key = executor_module._recorded_play_action_key_v1(recorded, 27)
    source = recorded.to_dict()
    source.pop("sha256")
    source["playActions"][30], source["playActions"][31] = (
        source["playActions"][31], source["playActions"][30]
    )
    assert executor_module._recorded_play_action_key_v1(
        ExactTracePrefixV1.build(source), 27,
    ) == key
    source["deckOrder"][0], source["deckOrder"][1] = (
        source["deckOrder"][1], source["deckOrder"][0]
    )
    assert executor_module._recorded_play_action_key_v1(
        ExactTracePrefixV1.build(source), 27,
    ) != key
    with pytest.raises(ValueError, match="outside the trace"):
        executor_module._recorded_play_action_key_v1(recorded, 32)


def test_tail_quality_escalation_divergence_is_candidate_miss_not_worker_failure(
    rules, complete_candidate, monkeypatch,
):
    raw = executor_module._construction_complete_prefix_v1(complete_candidate).to_dict()
    raw.pop("sha256")
    raw["escalationActions"] = [{"actor": "west", "action": "double"}]
    recorded = ExactTracePrefixV1.build(raw)
    source_raw = recorded.to_dict()
    source_raw.pop("sha256")
    source_raw.update(escalationActions=[], projectActions=[], projectDeclarations=[], playActions=[])
    source = ExactTracePrefixV1.build(source_raw)
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in ("bid", "escalation", "project", "play")}})
    monkeypatch.setattr(executor_module, "resolve_quality_escalation_continuation_v1",
        lambda *args, **kwargs: {"status": "complete", "partialReceipt": None,
            "finalPrefix": source.to_dict(), "initialNormalization": {}, "initialState": {},
            "projectPlayHandoff": {}, "artifacts": {}, "decisions": []})
    with pytest.raises(executor_module._QualityScenarioTargetMiss,
                       match="recorded escalation actions differ"):
        executor_module.resolve_quality_post_auction_continuation_v1(
            rules, source, {"kind": "escalation", "selector": {}}, policy,
            opportunity_order=("west", "east", "north", "south"),
            maximum_escalation_decisions=20, maximum_nodes=1000, maximum_depth=8,
            maximum_worlds=1, maximum_seconds=30, maximum_decision_seconds=10,
            maximum_evidence_bytes=1000, recorded_events=(), recorded_trace=recorded,
            continue_recorded_prefix=True, stop_on_recorded_quality_failure=True)


def test_campaign_progress_keeps_quality_failure_reason_without_private_evidence():
    summary = executor_module._campaign_incomplete_compilation_summary_v1({
        "stage": "playingQuality", "reason": "decision_point_target_miss",
        "qualityStatus": "quality_failed", "qualityFailureStage": "action_analysis",
        "qualityFailureReason": "recorded_action_not_optimal",
        "qualityEvidence": {"hiddenHands": ["SA"]},
    })
    assert summary == {
        "stage": "playingQuality", "reason": "decision_point_target_miss",
        "qualityStatus": "quality_failed", "qualityFailureStage": "action_analysis",
        "qualityFailureReason": "recorded_action_not_optimal",
    }


def test_observer_safe_policy_uses_scenario_alignment_only_between_equal_best_actions(
    monkeypatch, rules,
):
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "successThreshold": 0, "targetTeam": "west_east",
        "utility": "awarded_point_margin", "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    normalized = SimpleNamespace(
        decision_kind="play", next_actor="east",
        state={
            "contract": "sun", "trumpSuit": None,
            "currentTrick": {"index": 4, "leader": "north", "plays": [
                {"actor": "north", "card": "H10"},
                {"actor": "west", "card": "H7"},
                {"actor": "south", "card": "H8"},
            ]},
            "hands": {"east": ["HA", "C7"]},
        },
        legal_actions=({"code": "HA"}, {"code": "C7"}),
    )
    monkeypatch.setattr(executor_module, "_observer_safe_play_score_v1", lambda *args, **kwargs: 100)
    selected, artifact = executor_module._observer_safe_play_policy_artifact_v1(
        rules, normalized, request, actual_action=None,
        preferred_completed_trick_winner=Seat.EAST,
        maximum_seconds=1, cancel_check=None,
    )
    quality = artifact["proofModel"]["actionQuality"]
    assert selected == "HA"
    assert set(quality["optimalActions"]) == {"HA", "C7"}
    assert quality["regret"] == 0
    assert quality["selectionTieBreak"] == "scenario_target_actor_equal_value_v1"
    assert "west" not in artifact["allocationModel"]

    monkeypatch.setattr(
        executor_module, "_observer_safe_play_score_v1",
        lambda *args, **kwargs: 10 if kwargs.get("takes_control") else 20,
    )
    selected, artifact = executor_module._observer_safe_play_policy_artifact_v1(
        rules, normalized, request, actual_action=None,
        preferred_completed_trick_winner=Seat.EAST,
        maximum_seconds=1, cancel_check=None,
    )
    quality = artifact["proofModel"]["actionQuality"]
    assert selected == "C7"
    assert quality["optimalActions"] == ["C7"]
    assert quality["selectionTieBreak"] is None


def test_tactical_category_policy_rejects_losing_king_when_ace_takes_control(rules):
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "successThreshold": 0, "targetTeam": "west_east",
        "utility": "awarded_point_margin", "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    normalized = SimpleNamespace(
        decision_kind="play", next_actor="east",
        state={
            "contract": "sun", "trumpSuit": None,
            "currentTrick": {"index": 2, "leader": "north", "plays": [
                {"actor": "north", "card": "H10"},
                {"actor": "west", "card": "H7"},
                {"actor": "south", "card": "H8"},
            ]},
            "hands": {"east": ["HA", "HK"]},
        },
        legal_actions=({"code": "HA"}, {"code": "HK"}),
    )
    selected, artifact = executor_module._observer_safe_play_policy_artifact_v1(
        rules, normalized, request, actual_action=None,
        policy_id="observer_safe_tactical_category_v1",
        maximum_seconds=1, cancel_check=None,
    )
    quality = artifact["proofModel"]["actionQuality"]
    assert selected == "HA"
    assert quality["optimalActions"] == ["HA"]
    assert quality["actionScores"] == [
        {"action": "HA", "value": 5},
        {"action": "HK", "value": 3},
    ]


@pytest.mark.real_clock
@pytest.mark.parametrize("visibility", ["actor_information_set", "complete_state"])
def test_immediate_quality_readback_reuses_only_owned_normalizations(root, rules, monkeypatch, visibility):
    from baloot_platform.research import result_authority as authority
    prefix, template = _quality_continuation_input(root, "bounded" if visibility == "actor_information_set" else "exact")
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(raw)
    request_raw = template.to_dict()
    request_raw.pop("sha256")
    request_raw["visibility"] = visibility
    request = ScenarioProofRequestV1.build(request_raw)
    profile = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    calls, owned, verified = [], {}, set()
    normalize = OrToolsExactStateNormalizerV1.normalize
    normalize_auto = OrToolsExactStateNormalizerV1.normalize_play_or_project
    readback = executor_module._read_back_generated_quality_v1

    def tracked(self, *args, **kwargs):
        calls.append(args[0].sha256)
        return normalize(self, *args, **kwargs)

    def tracked_auto(self, *args, **kwargs):
        calls.append(args[0].sha256)
        return normalize_auto(self, *args, **kwargs)

    def immediate(value, remaining, **kwargs):
        owned.update(kwargs["verified_normalizations"])
        assert owned and all(state._play_template is None for state in owned.values())
        before = len(calls)
        result = readback(value, remaining, **kwargs)
        assert len(calls) == before  # No duplicate CP normalization in immediate readback.
        return result

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", tracked)
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", tracked_auto)
    monkeypatch.setattr(executor_module, "_read_back_generated_quality_v1", immediate)
    value = resolve_quality_play_continuation_v1(rules, prefix, request,
        _verified_artifacts=verified,
        proof_budget_profiles={"test": profile}, search_budget_profiles={"test": {"maximumConflicts": 250_000}},
        opportunity_order=("west", "east", "north", "south"), escalation_events=[],
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=64,
        maximum_seconds=120, maximum_decision_seconds=60, maximum_evidence_bytes=30_000_000)
    assert value["status"] == "complete", value.get("partialReceipt")
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, request=request, rules=rules,
        search_budget_profiles={"test": {"maximumConflicts": 250_000}},
        opportunity_order=("west", "east", "north", "south"), escalation_events=[])
    before = len(calls)
    authority._validate_native_decision_quality_evidence_v1(value, **context,
        verified_artifacts=frozenset(verified))
    assert len(calls) == before  # Later same-run admission keeps the completed readback.
    assert not any(token in json.dumps(value) for token in verified - set(owned))
    changed = copy.deepcopy(value)
    changed["decisions"][0]["selectionReason"] = "substituted"
    changed["decisions"][0]["sha256"] = canonical_sha256({key: item
        for key, item in changed["decisions"][0].items() if key != "sha256"})
    changed["sha256"] = canonical_sha256({key: item for key, item in changed.items()
        if key not in {"sha256", "executionReceipt"}})
    unchanged = set(verified)
    with pytest.raises(ValueError, match="quality (decision|action proof) binding mismatch"):
        authority._validate_native_decision_quality_evidence_v1(changed, **context,
            verified_artifacts=verified, verified_normalizations=owned)
    assert verified == unchanged
    for reason in ("cancelled", "maximum_seconds"):
        checks = []
        def stop_after_context():
            checks.append(True)
            return len(checks) > 1
        with monkeypatch.context() as patch:
            if reason == "maximum_seconds":
                def clock(*args):
                    def remaining():
                        if stop_after_context():
                            raise executor_module.InformationSetCompilationStopped(reason, stage="quality_readback")
                        return 1
                    return remaining
                patch.setattr(authority, "_quality_readback_clock", clock)
            with pytest.raises(executor_module.InformationSetCompilationStopped, match=reason):
                authority._validate_native_decision_quality_evidence_v1(value, **context,
                    verified_artifacts=verified, cancel_check=stop_after_context if reason == "cancelled" else None)
    before = len(calls)
    authority.validate_native_decision_quality_evidence_v1(value, **context)
    assert len(calls) > before  # External reads still reconstruct.
    with pytest.raises(TypeError, match="verified_normalizations"):
        authority.validate_native_decision_quality_evidence_v1(value, **context, verified_normalizations=owned)
    digest = next(iter(owned))
    terminal = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        ExactTracePrefixV1.build(value["finalPrefix"]), maximum_seconds=30)
    with pytest.raises(ValueError, match="normalization context substitution"):
        authority._validate_native_decision_quality_evidence_v1(value, **context,
            verified_artifacts=frozenset(owned), verified_normalizations={**owned, digest: terminal})
    with pytest.raises(ValueError, match="normalization has invalid type"):
        authority._validate_native_decision_quality_evidence_v1(value, **context,
            verified_artifacts=frozenset(owned), verified_normalizations={**owned, digest: object()})
    before = len(calls)
    authority._validate_native_decision_quality_evidence_v1(value, **context,
        verified_normalizations={key: object() for key in owned})
    assert len(calls) > before  # No verified digest means no reuse.
    with pytest.raises(executor_module.InformationSetCompilationStopped, match="cancelled"):
        authority._validate_native_decision_quality_evidence_v1(value, **context,
            verified_artifacts=frozenset(owned), verified_normalizations=owned, cancel_check=lambda: True)
    cold = set(owned)
    with monkeypatch.context() as patch:
        def reject_companion(*args, **kwargs):
            raise ValueError("companion rejected")
        patch.setattr(authority, "_validate_basic_policy_quality_companions_v1", reject_companion)
        with pytest.raises(ValueError, match="companion rejected"):
            authority._validate_native_decision_quality_evidence_v1(value, **context,
                verified_artifacts=cold, verified_normalizations=owned)
    assert cold == set(owned)  # A failed final companion cannot mint a stage token.


@pytest.mark.parametrize("field", ["core_hash", "rules", "play_domain", "source_prefix", "request",
    "policy", "execution_budget", "search_budget_profiles", "private_recall", "escalation_events", "opportunity_order"])
def test_quality_stage_token_binds_actual_context(rules, field):
    from baloot_platform.research.result_authority import _quality_stage_verification_token_v1
    rules = RuleBook(rules.data)
    def document(value):
        return SimpleNamespace(to_dict=lambda: {"value": value, "sha256": "unchanged"})
    context = dict(core_hash="receipt", rules=rules, source_prefix=document(1), request=document(1),
        policy=document(1), execution_budget=document(1), search_budget_profiles={"test": {"maximumConflicts": 1}},
        private_recall=None, escalation_events=[], opportunity_order=("west", "east", "north", "south"))
    original = _quality_stage_verification_token_v1(**context)
    if field == "rules":
        rules.data["play"]["first_leader"] = "changed"
    elif field == "play_domain":
        rules.play_domain = replace(rules.play_domain, ekka_enabled=not rules.play_domain.ekka_enabled)
    elif field in {"source_prefix", "request", "policy", "execution_budget"}:
        context[field] = document(2)  # An unchanged asserted checksum cannot hide a changed body.
    else:
        context[field] = {"core_hash": "different", "search_budget_profiles": {"test": {"maximumConflicts": 2}},
            "private_recall": {"north": ["a" * 64]}, "escalation_events": [{"actor": "north", "actionCode": "pass"}],
            "opportunity_order": ("east", "west", "north", "south")}[field]
    assert _quality_stage_verification_token_v1(**context) != original


@pytest.mark.parametrize("field", ["artifact", "rules", "source_prefix", "normalized", "request",
    "maximum_nodes", "maximum_depth", "maximum_worlds", "declared_maximum_seconds", "policy_search"])
def test_exact_observed_immediate_token_binds_context(field):
    from types import SimpleNamespace
    from baloot_platform.research.result_authority import _exact_observed_verification_token_v1
    context = dict(artifact={"proofModel": {"sha256": "a"}}, rules=SimpleNamespace(sha256="rules"),
        source_prefix=SimpleNamespace(sha256="prefix"), normalized=SimpleNamespace(to_dict=lambda: {"state": 1}),
        request=SimpleNamespace(sha256="request"), maximum_nodes=100, maximum_depth=8,
        maximum_worlds=64, declared_maximum_seconds=60, policy_search=None)
    original = _exact_observed_verification_token_v1(**context)
    if field in {"rules", "source_prefix", "request"}:
        context[field] = SimpleNamespace(sha256="different")
    elif field == "normalized":
        context[field] = SimpleNamespace(to_dict=lambda: {"state": 2})
    elif field == "artifact":
        context[field] = {"proofModel": {"sha256": "b"}}
    elif field == "policy_search":
        context[field] = "time_budgeted_symbolic_v1"
    else:
        context[field] += 1
    assert _exact_observed_verification_token_v1(**context) != original


def test_exact_featured_readback_reuses_only_fresh_context(root, rules, monkeypatch):
    from baloot_platform.research import result_authority as authority
    prefix, request = _quality_continuation_input(root, "exact")
    calls, captured, owned = [], {}, set()
    verify = authority.reverify_native_exact_observed_play_artifact_v1
    readback = executor_module._read_back_generated_quality_v1

    def tracked(artifact, **context):
        calls.append(canonical_sha256(artifact))
        return verify(artifact, **context)

    def immediate(result, remaining, **context):
        captured.update(context)
        return readback(result, remaining, **context)

    monkeypatch.setattr(authority, "reverify_native_exact_observed_play_artifact_v1", tracked)
    monkeypatch.setattr(executor_module, "_read_back_generated_quality_v1", immediate)
    result = resolve_quality_play_continuation_v1(rules, prefix, request,
        _verified_artifacts=owned,
        featured_play_index=len(prefix.play_actions), featured_decision_point={"kind": "play"},
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=64,
        maximum_seconds=180, maximum_decision_seconds=60, maximum_evidence_bytes=30_000_000)
    assert result["status"] == "complete", result.get("partialReceipt")
    featured = result["decisions"][0]["proofArtifactHash"]
    assert calls.count(featured) == 1  # Fresh reverify, not repeated by immediate readback.
    assert captured["verified_artifacts"] is owned and owned
    authority._validate_native_decision_quality_evidence_v1(result, **captured)
    assert calls.count(featured) == 1  # The outer admission owns the same fresh verification.
    context = {key: value for key, value in captured.items()
               if key not in {"verified_artifacts", "verified_normalizations"}}
    authority.validate_native_decision_quality_evidence_v1(result, **context)
    assert calls.count(featured) == 2  # Stored/public reads still independently reverify.
    cold_owned = set()
    for _ in range(2):
        authority._validate_native_decision_quality_evidence_v1(result, **context, verified_artifacts=cold_owned)
    assert cold_owned and calls.count(featured) == 3  # First cold check mints reuse only after success.
    assert not any(token in json.dumps(result) for token in owned)
    with pytest.raises(TypeError, match="verified_artifacts"):
        authority.validate_native_decision_quality_evidence_v1(result, **captured)
    altered = copy.deepcopy(result)
    artifact = altered["artifacts"].pop(featured)
    artifact["proofModel"]["claimSatisfied"] = not artifact["proofModel"]["claimSatisfied"]
    artifact["proof"]["identities"]["modelHash"] = canonical_sha256(artifact["proofModel"])
    digest = canonical_sha256(artifact)
    altered["artifacts"][digest] = artifact
    row = altered["decisions"][0]
    row["proofArtifactHash"] = digest
    row["sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
    altered["sha256"] = canonical_sha256({key: value for key, value in altered.items()
        if key not in {"sha256", "executionReceipt"}})
    before = set(owned)
    with pytest.raises(ValueError, match="fresh native proof"):
        authority._validate_native_decision_quality_evidence_v1(altered, **captured)
    assert owned == before  # An unsuccessful verification cannot mint a trusted token.
    assert digest in calls  # Old token cannot bless a replaced proof/model.


def test_featured_observed_rejection_preserves_proof_diagnostics(root, rules, monkeypatch):
    import yaml
    from baloot_platform.research import result_authority

    prefix, request = _quality_continuation_input(root, "exact")
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    monkeypatch.setattr(result_authority, "native_featured_play_proof_supported_v1", lambda artifact: False)
    result = resolve_quality_play_continuation_v1(rules, prefix, request,
        featured_play_index=len(prefix.play_actions), featured_decision_point={"kind": "play"},
        proof_budget_profiles={"test": profile}, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=64, maximum_seconds=180, maximum_decision_seconds=60,
        maximum_evidence_bytes=20_000_000)
    assert result["status"] == "incomplete" and result["decisions"] == []
    partial = result["partialReceipt"]
    assert partial["exhaustionReason"] == "featured_native_proof_unavailable"
    evidence = partial["evidence"]
    assert evidence["proofModelKind"] == "actor_information_set_cegis_model_v1"
    assert evidence["proofCompletionStatus"] == "exact_complete"
    assert evidence["rootActionCount"] > 1
    assert evidence["hiddenWorldCount"] > 0 and evidence["hiddenWorldCoverageComplete"] is True
    assert evidence["allocationReachability"] == "deal_auction_escalation_project_play_cp_sat_v1"


@pytest.mark.parametrize("play_index,seconds", [(23, 20), (24, 120)])
def test_tail_budget_keeps_initial_normalization_within_its_stage_budget(root, monkeypatch, play_index, seconds):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    prefix, request = _quality_continuation_input(root, "bounded")
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["playActions"] = raw["playActions"][:play_index]
    prefix = ExactTracePrefixV1.build(raw)
    settings = yaml.safe_load((root / "configs/challenge/actor_visible_tail_v3.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "budgeted",
        "visibility": "actor_information_set", "decisions": {kind: {
            "utility": request.utility, "budgetProfile": "bounded_v1",
            "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in SCENARIO_DECISION_KINDS}})
    class Captured(BaseException):
        pass
    def normalize(*args, **kwargs):
        assert kwargs["maximum_seconds"] == seconds
        raise Captured
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", normalize)
    set_monotonic(monkeypatch, executor_module, 0)
    with pytest.raises(Captured):
        resolve_quality_play_continuation_v1(rules, prefix, policy,
            maximum_nodes=2_000_000, maximum_depth=32, maximum_worlds=256, maximum_seconds=180,
            maximum_decision_seconds=120, maximum_evidence_bytes=20_000_000,
            execution_budget=ScenarioQualityExecutionBudgetV1.build(settings["qualityExecution"]),
            proof_budget_profiles=settings["proofBudgets"]["profiles"])


@pytest.mark.parametrize("policy_search,tail_budget", [
    (None, False), ("time_budgeted_symbolic_v1", False), ("time_budgeted_symbolic_v1", True)])
def test_exact_observed_quality_with_native_budget_completes_legal_suffix(root, rules, tmp_path, record_property, policy_search, tail_budget):
    import yaml
    prefix, request = _quality_continuation_input(root, "exact")
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    profile = settings["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    if policy_search is not None:
        profile["policySearch"] = policy_search
    assert profile["nativeObserved"]
    execution, policy = None, None
    profiles = {"test": profile}
    if tail_budget:
        policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2,
            "mode": "budgeted", "visibility": "actor_information_set",
            "decisions": {kind: {"utility": request.utility, "budgetProfile": "bounded_v1",
                "horizon": {"unit": "plies", "value": 1},
                "boundaryEvaluator": "current_raw_point_margin_v1"} for kind in SCENARIO_DECISION_KINDS}})
        execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
            "maximumWallSeconds": 180, "maximumEvidenceBytes": 20_000_000,
            "maximumEscalationDecisions": 20, "exactTailProof": True, "exactPlayFromTrick": 7,
            "earlyPlayPolicyId": "observer_safe_trick_control_v1", "exactTailBudgetProfile": "test"})
        profiles["bounded_v1"] = settings["proofBudgets"]["profiles"]["bounded_v1"]
        request = policy.request_for("play", "north")
    result = resolve_quality_play_continuation_v1(rules, prefix, policy or request,
        featured_play_index=len(prefix.play_actions), featured_decision_point={"kind": "play"},
        proof_budget_profiles=profiles, execution_budget=execution, maximum_nodes=100_000, maximum_depth=8,
        maximum_worlds=64, maximum_seconds=180, maximum_decision_seconds=60,
        maximum_evidence_bytes=20_000_000)
    receipt = tmp_path / "exact-observed-native-budget-suffix.json"
    receipt.write_text(json.dumps({"sourcePrefix": prefix.to_dict(), "rules": rules.data,
        "request": request.to_dict(), "proofBudget": profile, "result": result}, sort_keys=True), encoding="utf-8")
    record_property("observedExactSuffixEvidencePath", str(receipt))
    assert result["status"] == "complete", result.get("partialReceipt", {}).get("exhaustionReason")
    assert len(result["decisions"]) == 5
    assert result["proofBudgetProfiles"] == profiles
    assert result["finalPrefix"]["playActions"][:27] == prefix.to_dict()["playActions"]
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    for mutation in ("recall", "budget"):
        substituted = copy.deepcopy(result)
        if mutation == "recall":
            substituted["privateRecallHash"] = "0" * 64
        else:
            substituted["proofBudgetProfiles"]["test"]["maximumNodes"] -= 1
        substituted["sha256"] = canonical_sha256({key: value for key, value in substituted.items()
            if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="private recall mismatch|proof budget substitution"):
            validate_native_decision_quality_evidence_v1(substituted, source_prefix=prefix,
                rules_hash=rules.sha256, request=request, rules=rules,
                **({"policy": policy, "execution_budget": execution} if tail_budget else {}))
    for row in result["decisions"]:
        artifact = result["artifacts"][row["proofArtifactHash"]]
        assert artifact["proof"]["completionStatus"] == "exact_complete"
        assert artifact["request"]["mode"] == "exact" and "horizon" not in artifact["request"]
        assert row["actualActionOptimal"] is True and row["regret"] == 0
        if artifact["proofModel"]["kind"] != "actor_information_set_cegis_model_v1":
            continue
        from baloot_platform.research.result_authority import reverify_native_exact_observed_play_artifact_v1
        from baloot_platform.research.result_authority import native_featured_play_proof_supported_v1
        assert native_featured_play_proof_supported_v1(artifact)
        for field, value in (("complete", False), ("count", True), ("coverage", {})):
            unsupported = copy.deepcopy(artifact)
            unsupported["proof"]["hiddenWorlds"][field] = value
            assert not native_featured_play_proof_supported_v1(unsupported)
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["playActions"] = result["finalPrefix"]["playActions"][:row["playIndex"]]
        owned = ExactTracePrefixV1.build(raw)
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            owned, decision_point={"kind": "play"}, maximum_seconds=60)
        context = dict(rules=rules, source_prefix=owned, normalized=normalized,
            request=ScenarioProofRequestV1.build(artifact["request"]), maximum_nodes=100_000,
            maximum_depth=8, maximum_worlds=64, maximum_seconds=60, declared_maximum_seconds=60,
            policy_search=policy_search)
        assert artifact['rootWitnesses']
        assert {line['action'] for line in artifact['rootWitnesses']} == {
            root['action'] for root in artifact['proof']['rootActions']}
        reverify_native_exact_observed_play_artifact_v1(artifact, **context)
        historical = {key: value for key, value in artifact.items() if key != 'rootWitnesses'}
        reverify_native_exact_observed_play_artifact_v1(historical, **context)
        altered = copy.deepcopy(artifact)
        altered['rootWitnesses'][0]['worldId'] = 'foreign-world'
        with pytest.raises(ValueError, match='rootWitnesses'):
            reverify_native_exact_observed_play_artifact_v1(altered, **context)
        from dataclasses import replace
        with pytest.raises(ValueError, match="fresh native normalization"):
            reverify_native_exact_observed_play_artifact_v1(artifact, **{**context,
                "normalized": replace(normalized, project_declarations_verified=normalized.project_declarations_verified + 1)})
        for field in ("proofModel", "allocationModel", "proof", "graph"):
            with pytest.raises(ValueError, match="source or native coverage"):
                reverify_native_exact_observed_play_artifact_v1({**artifact, field: None}, **context)
        with pytest.raises(executor_module.InformationSetCompilationStopped, match="cancelled"):
            reverify_native_exact_observed_play_artifact_v1(artifact, **context, cancel_check=lambda: True)
        with pytest.raises(ValueError, match="finite declared limits"):
            reverify_native_exact_observed_play_artifact_v1(artifact, **{**context, "maximum_seconds": float("inf")})
        altered = copy.deepcopy(artifact)
        altered["graph"]["hiddenWorlds"]["complete"] = False
        altered["graph"]["sha256"] = canonical_sha256({key: value for key, value in altered["graph"].items() if key != "sha256"})
        altered["proof"]["identities"]["graphHash"] = altered["graph"]["sha256"]
        with pytest.raises(ValueError, match="fresh native reachability"):
            reverify_native_exact_observed_play_artifact_v1(altered, **context)
        altered = copy.deepcopy(artifact)
        altered["proofModel"]["claimSatisfied"] = not altered["proofModel"]["claimSatisfied"]
        altered["proof"]["identities"]["modelHash"] = canonical_sha256(altered["proofModel"])
        with pytest.raises(ValueError, match="fresh native proof"):
            reverify_native_exact_observed_play_artifact_v1(altered, **context)


@pytest.mark.parametrize("mode", ["exact", "bounded"])
def test_observed_selected_play_production_suffix(root, rules, tmp_path, record_property, mode):
    prefix, request = _quality_continuation_input(root, mode)
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw.update(projectDeclarations=[], projectAssertions=[], projectActions=[
        {"actor": seat.name.lower(), "action": "pass"}
        for seat in rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))])
    prefix = ExactTracePrefixV1.build(raw)
    profile = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        maximumWallSeconds=180, maximumCpuSeconds=180,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    result = resolve_quality_play_continuation_v1(rules, prefix, request,
        proof_budget_profiles={"test": profile}, search_budget_profiles={"test": {"maximumConflicts": 250_000}},
        opportunity_order=("west", "east", "north", "south"), escalation_events=[],
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=64,
        maximum_seconds=240, maximum_decision_seconds=180, maximum_evidence_bytes=30_000_000)
    receipt = tmp_path / "observed-selected-play-production.json"
    receipt.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    record_property("observedSelectedPlayProductionEvidencePath", str(receipt))
    assert result["status"] == "complete", result.get("partialReceipt")
    selected = [artifact for artifact in result["artifacts"].values()
        if artifact["proofModel"]["kind"] == (
            "native_bounded_play_quality_model_v1" if mode == "bounded" else "native_observed_play_quality_model_v1")]
    assert selected
    from baloot_platform.research.result_authority import native_featured_play_proof_supported_v1
    assert all(artifact["proof"]["winningRootActionCount"] is None
        and not native_featured_play_proof_supported_v1(artifact) for artifact in selected)
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, request=request, rules=rules,
        search_budget_profiles={"test": {"maximumConflicts": 250_000}},
        opportunity_order=("west", "east", "north", "south"), escalation_events=[])
    validate_native_decision_quality_evidence_v1(result, **context)
    assert all(artifact["proofModel"]["actionQuality"]["optimalActionCount"] is None for artifact in selected)
    with pytest.raises(ValueError, match="history substitution"):
        validate_native_decision_quality_evidence_v1(result, **{**context, "escalation_events": None})
    if mode == "bounded":
        # One ply really fits this last-card-of-trick fixture. Instead forge a
        # winning count, resealing every outer hash to exercise semantic readback.
        altered = copy.deepcopy(result)
        digest = next(key for key, artifact in altered["artifacts"].items()
            if artifact["proofModel"]["kind"] == "native_bounded_play_quality_model_v1")
        artifact = altered["artifacts"].pop(digest)
        artifact["proof"]["winningRootActionCount"] = 1
        artifact["proof"]["sha256"] = canonical_sha256({key: value for key, value in artifact["proof"].items()
            if key != "sha256"})
        replacement = canonical_sha256(artifact)
        altered["artifacts"][replacement] = artifact
        for row in altered["decisions"]:
            if row["proofArtifactHash"] == digest:
                row["proofArtifactHash"] = replacement
                row["sha256"] = canonical_sha256({key: value for key, value in row.items() if key != "sha256"})
        altered["sha256"] = canonical_sha256({key: value for key, value in altered.items()
            if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="bounded selected play artifact binding mismatch"):
            validate_native_decision_quality_evidence_v1(altered, **context)
    else:
        from baloot_platform.research.result_authority import validate_native_constant_escalation_budget_v1

        budget = ScenarioProofBudgetV1.build("test", profile)
        premise = selected[0]["graph"]["bestProof"]
        validate_native_constant_escalation_budget_v1(premise, budget, maximum_conflicts=250_000)
        for key, over_cap in (("iterations", 1001), ("queries", 2001), ("observations", 2001),
                ("conflictsPerSolve", 250_001), ("depth", 65), ("seconds", math.nextafter(180.0, math.inf))):
            altered_limits = {**premise["limits"], key: over_cap}
            match = "frozen execution limits" if key == "seconds" else "frozen proof limits"
            with pytest.raises(ValueError, match=match):
                validate_native_constant_escalation_budget_v1(
                    {**premise, "limits": altered_limits}, budget, maximum_conflicts=250_000)

        altered = copy.deepcopy(result)
        altered["limits"]["maximumDepth"] = 1
        altered["sha256"] = canonical_sha256({key: value for key, value in altered.items()
            if key not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="exceeds caller depth"):
            validate_native_decision_quality_evidence_v1(altered, **context)


def test_quality_play_selector_preserves_forced_early_and_native_priority(monkeypatch):
    calls = []
    current = object()
    request = SimpleNamespace(mode="bounded")
    rules = object()
    recall = {"south": ("H7",)}
    budget = SimpleNamespace(native_observed=object())
    policy = SimpleNamespace(mode="budgeted")
    execution = SimpleNamespace(early_play_policy_id="observer_safe_trick_control_v1",
                                exact_play_from_trick=7,
                                prune_own_project_pass_worlds=False)
    normalized = SimpleNamespace(decision_kind="play", legal_actions=({"code": "H7"},),
                                 state={"currentTrick": {"index": 0}})

    def forced(*args, **kwargs):
        calls.append(("forced", kwargs))
        return "H7", {"kind": "forced"}

    def early(*args, **kwargs):
        calls.append(("early", kwargs))
        return "H7", {"kind": "early"}

    class Native:
        def __init__(self, received_rules, *, native_cp_sat, prune_own_project_pass_worlds):
            assert received_rules is rules and native_cp_sat is True
            assert prune_own_project_pass_worlds is False

        def _prove_observed_play(self, *args, **kwargs):
            calls.append(("native", kwargs))
            return "proof", {"kind": "native"}

    monkeypatch.setattr(executor_module, "_forced_quality_artifact_v1", forced)
    monkeypatch.setattr(executor_module, "_observer_safe_play_policy_artifact_v1", early)
    monkeypatch.setattr(executor_module, "OrToolsActorInformationSetCompilerV1", Native)
    common = dict(quality_policy=policy, execution_budget=execution, budget=budget,
                  actual_action=None, remaining_seconds=lambda: 4.0,
                  cancel_check=None, private_recall=recall, maximum_worlds=3,
                  maximum_depth=8)

    artifact, proof, native, timing = executor_module._select_quality_play_action_v1(
        rules, current, normalized, request, featured_now=True, **common)
    assert (artifact, proof, native, timing) == ({"kind": "forced"}, None, None,
                                                  "forcedSelectionSeconds")

    normalized.legal_actions = ({"code": "H7"}, {"code": "HA"})
    artifact, proof, native, timing = executor_module._select_quality_play_action_v1(
        rules, current, normalized, request, featured_now=False, **common)
    assert (artifact, proof, native, timing) == ({"kind": "early"}, None, None,
                                                  "policySelectionSeconds")

    artifact, proof, native, timing = executor_module._select_quality_play_action_v1(
        rules, current, normalized, request, featured_now=True, **common)
    assert (artifact, proof, native, timing) == (None, "proof", {"kind": "native"},
                                                  "nativeExecutionSeconds")
    assert [kind for kind, _ in calls] == ["forced", "early", "native"]
    assert calls[-1][1]["project_prefix"] is current
    assert calls[-1][1]["private_recall"] is recall
    assert calls[-1][1]["budget"] is budget
    assert calls[-1][1]["maximum_worlds"] == 3
    assert calls[-1][1]["maximum_depth"] == 8
    request.mode = "exact"
    assert executor_module._select_quality_play_action_v1(
        rules, current, normalized, request, featured_now=True, **common) == (None, None, None, None)
    assert [kind for kind, _ in calls] == ["forced", "early", "native"]


@pytest.mark.parametrize("case", ["generated", "recorded", "missing_policy", "invalid_policy", "suboptimal", "incomplete", "cancelled"])
def test_observed_play_driver_owns_shared_budget_and_policy(root, monkeypatch, case):
    from baloot_platform.research import result_authority
    raw = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        maximumWallSeconds=10, maximumCpuSeconds=11,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 41,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    budget = ScenarioProofBudgetV1.build("test", raw)
    request = SimpleNamespace(mode="exact", visibility="actor_information_set",
        utility="awarded_point_margin", line_universe=None, budget_profile="test",
        selected_action="HA" if case == "recorded" else None)
    normalized = SimpleNamespace(next_actor="south", legal_actions=[{"code": "H7"}, {"code": "HA"}])
    calls = []
    # A clock base where (start + 10) - start rounds above the frozen allowance.
    clock = [math.nextafter(16.0, 0.0)]
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], process_time=lambda: clock[0]))

    class Native:
        def __init__(self, rules, *, native_cp_sat):
            assert native_cp_sat is True

        def _solve_native_post_auction_policy_bounds(self, *args, **kwargs):
            calls.append(kwargs)
            clock[0] += 2
            return {"status": "incomplete" if case == "incomplete" else "scoped_bound_converged",
                "sha256": "native-test-bound",
                "strategy": {"policy": {} if case == "missing_policy" else {"root": "invalid" if case == "invalid_policy" else "H7"},
                    "lowerBound": -1 if case == "suboptimal" and len(calls) == 2 else 0, "upperBound": 0,
                    **({"finalBranchProofs": [{"branchModelHash": "native-test"}]} if case == "generated" else {})},
                "preparation": {"rootObservationHash": "root"}}

    def build(selected, best, **kwargs):
        assert (selected is best) is (case == "generated")
        assert kwargs["budget"] is budget
        assert kwargs["escalation_events"] == []
        assert 0 < kwargs["maximum_seconds"] <= 10
        return {"verified": True}

    monkeypatch.setattr(executor_module, "OrToolsActorInformationSetCompilerV1", Native)
    monkeypatch.setattr(result_authority, "_build_native_observed_play_quality_artifact_v1", build)

    def run():
        prefix = SimpleNamespace(deck_order=tuple(range(32)),
            play_actions=tuple(range(27 if case == "recorded" else 4)))
        return executor_module._observed_play_quality_artifact_v1(object(), normalized, prefix, request,
            budget=budget, opportunity_order=("south", "west", "north", "east"),
            escalation_events=[], maximum_conflicts=1000, maximum_depth=32,
            maximum_seconds=10, cancel_check=lambda: case == "cancelled")

    if case in {"invalid_policy", "suboptimal", "incomplete", "cancelled"}:
        with pytest.raises(executor_module.InformationSetCompilationStopped):
            run()
        assert len(calls) == (0 if case == "cancelled" else 2 if case == "suboptimal" else 1)
    else:
        assert run() == {"verified": True}
        assert [call["selected_root_action"] for call in calls] == (
            [None] if case == "generated" else [None, "HA" if case == "recorded" else "H7"])
        assert all(call["maximum_queries"] == 10 and call["visibility"] == "actor_information_set" for call in calls)
        assert all(call["observed_play_bounds_version"] == 2
            and call["path_floor_version"] == 2 for call in calls)
        assert all(call["compact_public_ownership"] is True for call in calls)
        assert all(call["maximum_seconds"] == 10 for call in calls)
        assert [call["runtime_maximum_seconds"] for call in calls] == pytest.approx([10, 8][:len(calls)])


@pytest.mark.real_clock
@pytest.mark.parametrize("stage", ["construction", "verification", "after_construction"])
@pytest.mark.parametrize("reason", ["maximum_seconds", "maximum_cpu_seconds", "cancelled", "maximum_queries"])
def test_observed_quality_stop_keeps_deadline_and_cancellation_distinct(root, monkeypatch, stage, reason):
    from baloot_platform.research import result_authority
    budget = ScenarioProofBudgetV1.build("test", _quality_budget_profile(root,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 41,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000}))
    clock = [0, 0]
    cancelled = [False]
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(
        monotonic=lambda: clock[0], process_time=lambda: clock[1]))
    cause = {}

    def stop(*, raise_error=True):
        clock[0 if reason != "maximum_cpu_seconds" else 1] = 11
        cancelled[0] = reason == "cancelled"
        error = executor_module.InformationSetCompilationStopped(
            "maximum_queries" if reason == "maximum_queries" else "cancelled",
            stage="native-test", evidence={"queriesUsed": 7})
        cause.update(error.partial_receipt)
        if raise_error:
            raise error

    class Native:
        def __init__(self, *args, **kwargs):
            pass

        def _solve_native_post_auction_policy_bounds(self, *args, **kwargs):
            if stage == "construction":
                stop()
            if stage == "after_construction":
                stop(raise_error=False)
            return {"status": "scoped_bound_converged", "strategy": {
                "policy": {"root": "H7"}, "lowerBound": 0, "upperBound": 0},
                "preparation": {"rootObservationHash": "root"}, "sha256": "native-test-bound"}

    monkeypatch.setattr(executor_module, "OrToolsActorInformationSetCompilerV1", Native)
    monkeypatch.setattr(result_authority, "_build_native_observed_play_quality_artifact_v1",
        lambda *args, **kwargs: stop())
    request = SimpleNamespace(mode="exact", visibility="actor_information_set",
        utility="awarded_point_margin", line_universe=None, budget_profile="test", selected_action=None)
    with pytest.raises(executor_module.InformationSetCompilationStopped) as caught:
        executor_module._observed_play_quality_artifact_v1(object(),
            SimpleNamespace(next_actor="south", legal_actions=[{"code": "H7"}]),
            SimpleNamespace(deck_order=tuple(range(32)), play_actions=tuple(range(4))), request,
            budget=budget, opportunity_order=("south", "west", "north", "east"),
            escalation_events=[], maximum_conflicts=1000, maximum_depth=32,
            maximum_seconds=10, cancel_check=lambda: cancelled[0])
    receipt = caught.value.partial_receipt
    assert receipt["exhaustionReason"] == (
        "maximum_seconds" if stage == "after_construction" and reason == "maximum_queries" else reason)
    if stage == "after_construction":
        assert receipt["stage"] == "observed_play_quality"
    elif reason in {"maximum_seconds", "maximum_cpu_seconds"}:
        assert receipt["evidence"]["causeReceipt"] == cause
    else:
        assert receipt == cause


def test_observer_safe_play_score_branches_and_action_tie_break():
    from baloot_platform.research.ortools_play_tail import _observer_safe_play_score_v1

    assert _observer_safe_play_score_v1(7, 11) == 20_711
    assert _observer_safe_play_score_v1(7, 11, partner_winning=True, takes_control=False) == 38_893
    assert _observer_safe_play_score_v1(7, 11, partner_winning=True, takes_control=False, feed_partner=True) == 41_093
    assert _observer_safe_play_score_v1(7, 11, partner_winning=False, takes_control=True, feed_partner=True) == 49_289
    assert _observer_safe_play_score_v1(7, 11, partner_winning=False, takes_control=True) == 49_289
    assert _observer_safe_play_score_v1(7, 11, partner_winning=False, takes_control=False) == 28_893
    assert _observer_safe_play_score_v1(7, 11, partner_winning=True, takes_control=True) == 9_289

    score = _observer_safe_play_score_v1(7, 11)
    assert score * 100 - 3 > score * 100 - 35  # normal play beats Ekka on a score tie
    with pytest.raises(ValueError, match="both trick-control inputs"):
        _observer_safe_play_score_v1(7, 11, partner_winning=True)
    with pytest.raises(ValueError, match="feed selection must be boolean"):
        _observer_safe_play_score_v1(7, 11, feed_partner=1)


def test_play_zero_observation_hash_ignores_hidden_opponent_hands(
    complete_candidate,
) -> None:
    first = executor_module._candidate_play_zero_observation_hash_v1(
        complete_candidate
    )
    trace = copy.deepcopy(complete_candidate.trace)
    hands = trace["play"]["initialHands"]
    actor = trace["play"]["plays"][0]["actor"]
    hidden = [seat for seat in hands if seat != actor]
    hands[hidden[0]], hands[hidden[1]] = hands[hidden[1]], hands[hidden[0]]

    assert executor_module._candidate_play_zero_observation_hash_v1(
        SimpleNamespace(trace=trace)
    ) == first

@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["cancelled", "maximum_seconds", "maximum_evidence_bytes",
                                  "escalation_incomplete", "quality_failed", "complete"])
@pytest.mark.parametrize("runtime", [None, 10])
def test_post_auction_handoff_preserves_budgets_and_stage_evidence(root, rules, monkeypatch, stop, runtime):
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin", "budgetProfile": "test",
            "horizon": {"unit": "plies", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"}
            for kind in ("bid", "escalation", "project", "play")}})
    clock, calls, checkpoints, verified = [0], [], [], set()
    search_profiles = {"test": {"maximumConflicts": 250_000}}
    def final_normalized_callback(_value):
        return None
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    stage = {"status": "incomplete" if stop == "escalation_incomplete" else "complete",
        "partialReceipt": {"reason": "retained"} if stop == "escalation_incomplete" else None,
        "finalPrefix": prefix.to_dict(), "initialNormalization": {}, "initialState": {},
        "projectPlayHandoff": {}, "artifacts": {},
        "events": [{"actor": "west", "actionCode": "decline"}]}
    def escalation(*args, **kwargs):
        assert kwargs["runtime_maximum_seconds"] == (30 if runtime is None else runtime)
        clock[0] = 31 if stop == "maximum_seconds" else 2
        return stage
    def play(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["_verified_artifacts"] is verified
        assert kwargs["final_normalized_callback"] is final_normalized_callback
        assert kwargs["runtime_maximum_seconds"] == (28 if runtime is None else runtime - 2)
        assert kwargs["maximum_seconds"] == 30
        assert kwargs["escalation_events"] == stage["events"]
        assert kwargs["opportunity_order"] == ("west", "east", "north", "south")
        assert kwargs["search_budget_profiles"] is search_profiles
        assert kwargs["maximum_evidence_bytes"] == 994  # Three retained empty JSON objects.
        return {"status": stop, "partialReceipt": None, "finalPrefix": prefix.to_dict(),
                "executionReceipt": {"evidenceBytes": 1, "elapsedSeconds": 0}}
    monkeypatch.setattr(executor_module, "resolve_quality_escalation_continuation_v1", escalation)
    monkeypatch.setattr(executor_module, "resolve_quality_play_continuation_v1", play)
    result = executor_module.resolve_quality_post_auction_continuation_v1(rules, prefix,
        {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
        opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
        maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=30,
        maximum_decision_seconds=10, maximum_evidence_bytes=6 if stop == "maximum_evidence_bytes" else 1000,
        cancel_check=lambda: stop == "cancelled", runtime_maximum_seconds=runtime,
        checkpoint_callback=checkpoints.append,
        _verified_artifacts=verified,
        final_normalized_callback=final_normalized_callback, search_budget_profiles=search_profiles)
    assert result["escalation"] == stage
    assert result["finalPrefix"] == prefix.to_dict()
    assert result["scope"] == "post_auction_only"
    assert len(calls) == (1 if stop in {"complete", "quality_failed"} else 0)
    assert result["status"] == (stop if calls else "incomplete")
    if stop in {"cancelled", "maximum_seconds", "maximum_evidence_bytes"}:
        assert result["partialReceipt"]["exhaustionReason"] == stop
    assert result["executionReceipt"]["evidenceBytes"] == (7 if calls else 6)
    assert checkpoints == [{"schemaVersion": 1, "kind": "quality_stage_checkpoint_v1",
        "rulesHash": rules.sha256, "stage": "escalation",
        "elapsedSeconds": 31 if stop == "maximum_seconds" else 2}]


@pytest.mark.real_clock
@pytest.mark.parametrize("case", ["miss", "forced", "forced_unbound", "forced_incomplete",
    "pass", "missing", "incomplete", "timeout", "readback_failed", "extra_row", "before_exact", "disabled"])
@pytest.mark.parametrize("at_exact_start", [False, True])
def test_curated_focal_precheck_rejects_only_complete_native_interest_miss(
    root, rules, monkeypatch, case, at_exact_start,
):
    from types import SimpleNamespace
    from test_research_decision_selection_qualification import _actor_interest_fixture

    recorded, request = _quality_continuation_input(root, "exact")
    raw = recorded.to_dict()
    raw.pop("sha256", None)
    raw.update(projectActions=[], stateAssertions={})
    if at_exact_start:
        raw["playActions"] = raw["playActions"][:24]
    if case == "before_exact":
        raw["playActions"] = raw["playActions"][:23]
    recorded = ExactTracePrefixV1.build(raw)
    source = ExactTracePrefixV1.build({**raw, "playActions": [], "projectDeclarations": []})
    analysis = {key: request.to_dict()[key] for key in ("utility", "budgetProfile")}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {
            kind: analysis for kind in ("bid", "escalation", "project", "play")}})
    interest, fixture = _actor_interest_fixture()
    group = fixture["postAuction"]["projectPlay"]
    group["decisions"][0]["playIndex"] = len(recorded.play_actions)
    if case in {"miss", "readback_failed", "extra_row"}:
        for row in group["artifacts"]["proof"]["proof"]["rootActions"]:
            row["value"] = -10
    if case.startswith("forced"):
        action = group["decisions"][0]["action"]
        artifact = group["artifacts"]["proof"]
        artifact["proof"].pop("hiddenWorlds")
        artifact["proof"].update(completionScope="action_optimality_only", solverStatus="optimal",
            selectedAction=action, rootActions=[{"action": action, "value": None}])
        artifact["graph"] = {"kind": "forced_legal_action_graph_v1", "legalActions": [action]}
        artifact["proofModel"] = {"kind": "forced_action_optimality_model_v1"}
        if case == "forced_unbound":
            artifact["graph"]["legalActions"].append("C7")
    if case == "missing":
        group.pop("baselineEvidence")
    execution = SimpleNamespace(exact_tail_proof=True, exact_play_from_trick=7, exact_tail_budget_profile=None,
        video_interestingness=interest, resolve_hidden_world_cap=lambda cap, **kwargs: cap)
    clock, calls, verified, checkpoints = [0.0], [], set(), []
    escalation = {"status": "complete", "partialReceipt": None, "finalPrefix": source.to_dict(),
        "initialNormalization": {}, "initialState": {}, "projectPlayHandoff": {}, "artifacts": {},
        "events": [{"actor": "west", "actionCode": "decline"}],
        "privateRecall": {"west": ["private-decline-window"]}}
    def escalations(*args, **kwargs):
        clock[0] += 2
        return escalation
    def play(_rules, prefix, _policy, **kwargs):
        calls.append(prefix)
        assert kwargs["private_recall"] is escalation["privateRecall"]
        assert kwargs["escalation_events"] is escalation["events"]
        assert kwargs["opportunity_order"] == ("west", "east", "north", "south")
        assert kwargs["_verified_artifacts"] is verified
        if prefix == recorded:
            assert kwargs["_curated_focal_only"] is True
            assert "recorded_trace" not in kwargs and "featured_play_index" not in kwargs
            assert kwargs["checkpoint_callback"] == checkpoints.append
            assert "final_normalized_callback" not in kwargs
            assert kwargs["runtime_maximum_seconds"] == 28
            assert kwargs["maximum_evidence_bytes"] == 994
            clock[0] += 3
            if case == "timeout":
                raise executor_module.InformationSetCompilationStopped("maximum_seconds", stage="quality_readback")
            marker = {"stage": "curated_focal_precheck", "reason": "focal_decision_complete"}
            if case in {"incomplete", "forced_incomplete"}:
                marker = {"stage": "proof", "exhaustionReason": "maximum_seconds"}
            if case == "readback_failed":
                marker = {"stage": "quality_readback", "exhaustionReason": "maximum_seconds",
                    "evidence": {"generationReceipt": marker}}
            if case == "extra_row":
                group["decisions"].append(copy.deepcopy(group["decisions"][0]))
            return {**group, "status": "incomplete", "partialReceipt": marker,
                "executionReceipt": {"evidenceBytes": 5, "elapsedSeconds": 3}}
        assert kwargs["recorded_trace"] == recorded and kwargs["continue_recorded_prefix"] is True
        assert kwargs["runtime_maximum_seconds"] == (28 if case in {"before_exact", "disabled"} else 25)
        assert kwargs["maximum_evidence_bytes"] == (994 if case in {"before_exact", "disabled", "timeout"} else 989)
        return {"status": "complete", "partialReceipt": None, "finalPrefix": source.to_dict(),
            "executionReceipt": {"evidenceBytes": 1, "elapsedSeconds": 0}}
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(executor_module, "resolve_quality_escalation_continuation_v1", escalations)
    monkeypatch.setattr(executor_module, "resolve_quality_play_continuation_v1", play)
    def run():
        return executor_module.resolve_quality_post_auction_continuation_v1(rules, source,
            {"kind": "escalation", "selector": {}}, policy, opportunity_order=("west", "east", "north", "south"),
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=64, maximum_seconds=30,
            maximum_decision_seconds=10, maximum_escalation_decisions=20, maximum_evidence_bytes=1000,
            recorded_events=[], recorded_trace=recorded, continue_recorded_prefix=True,
            stop_on_recorded_quality_failure=True, execution_budget=execution,
            featured_play_index=len(recorded.play_actions), _verified_artifacts=verified,
            checkpoint_callback=checkpoints.append,
            _curated_focal_precheck=case != "disabled")
    if case in {"miss", "forced"}:
        with pytest.raises(executor_module._QualityEditorialMiss) as exc:
            run()
        assert calls == [recorded]  # No earlier-tail audit or winner/tie companion work.
        assert exc.value.evidence["researchClaimStatus"] == "not_evaluated"
        assert ("interestingness_forced_legal_action" if case == "forced" else
                "interestingness_next_best_regret_insufficient") in exc.value.evidence["blockers"]
    else:
        result = run()
        assert result["status"] == "complete"  # Only the existing full stage supplies this status.
        assert calls == ([source] if case in {"before_exact", "disabled"} else [recorded, source])
    diagnostics = [row for row in checkpoints if row.get("stage") == "curated_focal_precheck"]
    if case in {"before_exact", "disabled"}:
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1
        assert diagnostics[0]["status"] == "incomplete"
        assert diagnostics[0]["elapsedSeconds"] == 3
        assert diagnostics[0]["partialReceipt"]["stage"] == (
            "quality_readback" if case in {"timeout", "readback_failed"}
            else "proof" if case in {"incomplete", "forced_incomplete"} else "curated_focal_precheck")


@pytest.mark.real_clock
@pytest.mark.parametrize("interruption", [None, "proof", "readback", "model_build"])
def test_curated_focal_only_retains_one_freshly_verified_native_decision(root, rules, monkeypatch, interruption):
    from baloot_platform.research import result_authority as authority
    from test_research_decision_selection_qualification import _actor_interest_fixture

    _, request = _quality_continuation_input(root, "exact")
    # Two worlds and a 34-point primary gap; baseline agreement still rejects full interest.
    prefix = ExactTracePrefixV1.build(json.loads(
        (root / "tests/fixtures/curated_primary_potential_passer.json").read_text()))
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    analysis = {key: request.to_dict()[key] for key in ("utility", "budgetProfile")}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: analysis for kind in SCENARIO_DECISION_KINDS}})
    interest, _ = _actor_interest_fixture()
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 120, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20, "maximumHiddenWorldsPerDecision": 64,
        "exactTailProof": True, "exactPlayFromTrick": 7,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1", "videoInterestingness": interest})
    compiled, reverified, cancelled = [], [], [False]
    compile_native = executor_module.compile_observed_decision_v1
    verify_native = authority.reverify_native_exact_observed_play_artifact_v1
    def compile_one(*args, **kwargs):
        compiled.append(len(args[2].play_actions))
        return compile_native(*args, **kwargs)
    def verify_one(artifact, **context):
        reverified.append(len(context["source_prefix"].play_actions))
        if interruption == "readback":
            cancelled[0] = True
        return verify_native(artifact, **context)
    def checkpoint(value):
        if interruption == "proof" and value.get("kind") == "quality_proof_checkpoint_v1":
            cancelled[0] = True
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", compile_one)
    monkeypatch.setattr(authority, "reverify_native_exact_observed_play_artifact_v1", verify_one)
    if interruption == "model_build":
        def stop_build(*args, **kwargs):
            raise PlayTailSolveStopped("unknown", evidence={"cancelled": True})
        monkeypatch.setattr(executor_module.OrToolsPlayTailStageV1, "build_model", stop_build)
    result = resolve_quality_play_continuation_v1(rules, prefix, policy,
        execution_budget=execution, _curated_focal_only=True,
        proof_budget_profiles={"test": _quality_budget_profile(root, maximumHiddenWorlds=64)},
        maximum_nodes=100_000, maximum_depth=8, maximum_worlds=64,
        maximum_seconds=120, maximum_decision_seconds=60, maximum_evidence_bytes=30_000_000,
        cancel_check=lambda: cancelled[0], checkpoint_callback=checkpoint)
    assert result["status"] == "incomplete"
    if interruption == "model_build":
        assert result["partialReceipt"]["exhaustionReason"] == "unknown"
        assert result["decisions"] == [] and result["artifacts"] == {}
        assert compiled == [] and reverified == []
        return
    if interruption is not None:
        assert result["partialReceipt"]["exhaustionReason"] == "cancelled"
        assert result["partialReceipt"].get("reason") != "focal_decision_complete"
        assert len(result["decisions"]) == (0 if interruption == "proof" else 1)
        assert set(compiled) == {len(prefix.play_actions)}
        return
    assert result["partialReceipt"] == {"stage": "curated_focal_precheck", "reason": "focal_decision_complete"}
    assert len(result["decisions"]) == 1
    assert result["coveredFromPlayIndex"] == len(prefix.play_actions)
    assert result["coveredToPlayIndex"] == len(prefix.play_actions) + 1
    assert len(result["finalPrefix"]["playActions"]) == result["coveredToPlayIndex"]
    assert compiled == [len(prefix.play_actions)] * 2  # Generation and fresh readback.
    assert reverified == [len(prefix.play_actions)]
    artifact = result["artifacts"][result["decisions"][0]["proofArtifactHash"]]
    assert artifact["proof"]["completionStatus"] == "exact_complete"
    assert artifact["proof"]["hiddenWorlds"]["count"] == 2
    assert {row["action"]: row["value"] for row in artifact["proof"]["rootActions"]} == {"CA": -10, "CK": -44}
    assert result["executionReceipt"]["readbackSeconds"] >= 0


@pytest.mark.parametrize("miss", ["gap", "worlds", "bytes", "cancel"])
def test_curated_primary_miss_skips_followup_but_cannot_admit_evidence(root, rules, monkeypatch, miss):
    from test_research_decision_selection_qualification import _actor_interest_fixture

    prefix, request = _quality_continuation_input(root, "exact")
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = "buyer_makes"
    rules = RuleBook(data)
    analysis = {key: request.to_dict()[key] for key in ("utility", "budgetProfile")}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: analysis for kind in SCENARIO_DECISION_KINDS}})
    interest, _ = _actor_interest_fixture()
    interest["minimumNextBestRegret" if miss == "gap" else "minimumConsistentWorlds"] = (
        100000 if miss == "gap" else 256)
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 120, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20, "maximumHiddenWorldsPerDecision": 64,
        "exactTailProof": True, "exactPlayFromTrick": 7,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1", "videoInterestingness": interest})
    owned, cancelled = {}, [False]
    prove = executor_module.prove_actor_information_set_v1

    def capture(compilation, request, **kwargs):
        proof = prove(compilation, request, **kwargs)
        owned.update(compilation=compilation, proof=proof, request=request)
        cancelled[0] = miss == "cancel"
        return proof

    def forbidden(*args, **kwargs):
        pytest.fail("a definite primary miss must skip baseline, transition and readback")

    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", capture)
    monkeypatch.setattr(executor_module, "_observer_safe_play_policy_artifact_v1", forbidden)
    if miss not in {"bytes", "cancel"}:
        monkeypatch.setattr(executor_module, "_read_back_generated_quality_v1", forbidden)
    verified = set()
    def run():
        return resolve_quality_play_continuation_v1(rules, prefix, policy, execution_budget=execution,
            _curated_focal_only=True, _verified_artifacts=verified,
            proof_budget_profiles={"test": _quality_budget_profile(root)},
            maximum_nodes=100_000, maximum_depth=8, maximum_worlds=64,
            maximum_seconds=120, maximum_decision_seconds=60,
            maximum_evidence_bytes=1 if miss == "bytes" else 30_000_000,
            cancel_check=lambda: cancelled[0])
    if miss in {"bytes", "cancel"}:
        result = run()
        assert result["status"] == "incomplete"
        assert result["partialReceipt"]["exhaustionReason"] == (
            "maximum_evidence_bytes" if miss == "bytes" else "cancelled")
        assert result["decisions"] == []
        assert verified == set()
        return
    with pytest.raises(executor_module._QualityEditorialMiss) as error:
        run()
    evidence = error.value.evidence
    assert evidence["verificationStatus"] == "builder_preview_only"
    assert evidence["researchClaimStatus"] == "not_evaluated"
    assert evidence["sourcePrefix"] == prefix.to_dict()
    assert ("interestingness_next_best_regret_insufficient" if miss == "gap" else
            "interestingness_consistent_worlds_insufficient") in evidence["blockers"]
    assert verified == set()
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "play"})
    artifact = evidence["primaryArtifact"]
    context = {**owned, "normalized": normalized, "rules": rules, "policy": interest}
    assert executor_module._curated_primary_interest_failures_v1(artifact, **context) == evidence["blockers"]
    other_team = "west_east" if artifact["request"]["targetTeam"] == "north_south" else "north_south"
    for field, change in (("request", {**artifact["request"], "targetTeam": other_team}),
            ("proof", {**artifact["proof"], "completionStatus": "incomplete"}),
            ("proof", {**artifact["proof"], "rootActions": [None, None]}),
            ("allocationModel", {**artifact["allocationModel"], "exactStateId": "STATE-OTHER"}),
            ("proofModel", {**artifact["proofModel"], "modelStatus": "quarantined"})):
        assert executor_module._curated_primary_interest_failures_v1({**artifact, field: change}, **context) == []


@pytest.mark.parametrize("scope", ["flag", "no_policy", "no_budget", "no_tail", "no_interest",
    "before_tail", "complete_state", "utility"])
def test_curated_focal_only_rejects_invalid_scope_before_normalization(root, rules, monkeypatch, scope):
    from test_research_decision_selection_qualification import _actor_interest_fixture

    prefix, request = _quality_continuation_input(root, "exact")
    data = copy.deepcopy(rules.data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = "buyer_makes"
    rules = RuleBook(data)
    if scope == "before_tail":
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["playActions"] = list(prefix.play_actions[:23])
        prefix = ExactTracePrefixV1.build(raw)
    interest, _ = _actor_interest_fixture()
    analysis = {key: request.to_dict()[key] for key in ("utility", "budgetProfile")}
    if scope == "utility":
        analysis["utility"] = "contract_success"
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "complete_state" if scope == "complete_state" else "actor_information_set",
        "decisions": {kind: analysis for kind in SCENARIO_DECISION_KINDS}})
    execution = ScenarioQualityExecutionBudgetV1.build({"schemaVersion": 1,
        "maximumWallSeconds": 30, "maximumEvidenceBytes": 10000, "maximumEscalationDecisions": 20,
        "maximumHiddenWorldsPerDecision": 64, "exactTailProof": scope != "no_tail",
        "exactPlayFromTrick": 7, "earlyPlayPolicyId": "observer_safe_trick_control_v1",
        **({"videoInterestingness": interest} if scope != "no_interest" else {})})
    monkeypatch.setattr(executor_module, "OrToolsExactStateNormalizerV1",
        lambda *args, **kwargs: pytest.fail("invalid focal scope reached normalization"))
    with pytest.raises(TypeError if scope == "flag" else ValueError, match="curated.focal"):
        resolve_quality_play_continuation_v1(rules, prefix, request if scope == "no_policy" else policy,
            execution_budget=None if scope == "no_budget" else execution,
            _curated_focal_only=1 if scope == "flag" else True,
            maximum_nodes=100, maximum_depth=8, maximum_worlds=64,
            maximum_seconds=30, maximum_decision_seconds=10, maximum_evidence_bytes=10000)


@pytest.mark.parametrize("mutation,reason", [("score", "source context"), ("deck", "source context"),
    ("short", "complete 32-play"), ("project_format", "project action semantics"),
    ("missing_events", "requires both"), ("missing_trace", "requires both")])
def test_post_auction_rejects_recorded_context_before_solver(root, rules, monkeypatch, mutation, reason):
    source, request = _quality_continuation_input(root, "bounded")
    recorded = source.to_dict()
    recorded.pop("sha256")
    recorded["projectActions"] = []
    recorded["playActions"].extend({"card": card, "ekka": False} for card in ("SA", "C9", "DA", "D8", "S7"))
    raw = {**recorded, "playActions": [], "stateAssertions": {}}
    prefix = ExactTracePrefixV1.build(raw)
    if mutation == "score":
        recorded["matchScore"] = [1, 0]
    elif mutation == "deck":
        recorded["deckOrder"] = list(reversed(recorded["deckOrder"]))
    elif mutation == "short":
        recorded["playActions"].pop()
    elif mutation == "project_format":
        recorded.pop("projectActions")
    analysis = {key: request.to_dict()[key] for key in ("utility", "budgetProfile", "horizon", "boundaryEvaluator")}
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: analysis for kind in ("bid", "escalation", "project", "play")}})
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid recording reached escalation analysis")
    monkeypatch.setattr(executor_module, "resolve_quality_escalation_continuation_v1", forbidden)
    with pytest.raises(ValueError, match=reason):
        executor_module.resolve_quality_post_auction_continuation_v1(rules, prefix,
            {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
            opportunity_order=("west", "east", "north", "south"), maximum_escalation_decisions=20,
            maximum_nodes=1000, maximum_depth=8, maximum_worlds=1, maximum_seconds=30,
            maximum_decision_seconds=10, maximum_evidence_bytes=1000,
            recorded_trace=None if mutation == "missing_trace" else ExactTracePrefixV1.build(recorded),
            recorded_events=None if mutation == "missing_events" else [])


def test_search_optional_project_prefix_rebinds_native_interleaved_root(
    root: Path,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    authored_prefix = _optional_project_prefix_v1()
    decision_point = {
        "kind": "project",
        "selector": {"owner": "east", "projectIndex": 0},
    }
    source = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(
        authored_prefix,
        decision_point=decision_point,
        maximum_seconds=15,
    )
    receipt = source.normalization_receipt.to_dict()
    projects = copy.deepcopy(receipt["derivedProjects"])
    projects["declarationActions"].append(
        {key: value for key, value in source.legal_actions[-1].items() if key != "code"}
    )
    semantic_hash = canonical_sha256({"kind": "search-project-fixture"})
    candidate = OrToolsNeutralCandidateV1(
        "search",
        {
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auction": {
                "completion": {
                    "actions": [{"actor": "north", "action": "sun"}]
                }
            },
            "escalation": receipt["derivedEscalation"],
            "projects": projects,
            "play": receipt["derivedPlay"],
        },
        {},
        tuple(range(32)),
        {
            "fixture": OrToolsResearchStageReceiptV1(
                "fixture",
                semantic_hash,
                semantic_hash,
                semantic_hash,
                "optimal",
            )
        },
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-SEARCH-PROJECT-ROOT",
                "state.phase",
                {"state": "playing"},
                "report",
            ),
            "decisionPoint": decision_point,
            "outcome": _leaf(
                "PRED-SEARCH-PROJECT-REPORT",
                "state.phase",
                {"state": "playing"},
                "report",
            ),
        }
    )

    rebound = executor_module._search_decision_prefix(candidate, scenario)

    assert [row["actor"] for row in rebound.project_actions] == [
        "north",
        "west",
        "south",
    ]
    assert [row["actor"] for row in rebound.play_actions] == [
        "north",
        "west",
        "south",
    ]
    normalized = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(
        rebound,
        decision_point=decision_point,
        maximum_seconds=15,
    )
    assert [row["code"].split(":", 1)[0] for row in normalized.legal_actions] == [
        "pass",
        "declare",
    ]


@pytest.mark.parametrize("hardness", ["hard", "report", "prefer"])
def test_native_project_tactic_dispatch_requests_paired_graph_for_every_hardness(root, monkeypatch, hardness):
    from baloot_platform.tactics import TacticalDetectorProfileV1

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = _optional_project_prefix_v1()
    decision = {"kind": "project", "selector": {"owner": "east", "projectIndex": 0}}
    called = []

    def compile_pair(_rules, normalized, _prefix, _decision, request, **options):
        # Real prefix normalization; tiny terminal stubs isolate dispatch, not round optimality.
        assert options["project_counterfactual"] is True and options["native_cp_sat"] is True
        assert normalized.next_actor == "east"
        codes = [row["code"] for row in normalized.legal_actions]
        assert codes[0] == "pass" and len(codes) == 2
        node = ProofGraphNodeV1.decision("root", "west_east", tuple(
            ProofGraphEdgeV1.build(index, code, f"end-{index}") for index, code in enumerate(codes)),
            actor_seat="east")
        def graph(neutral):
            return ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(node,
                ProofGraphNodeV1.leaf("end-0", -6), ProofGraphNodeV1.leaf("end-1", -6 if neutral else 6)))
        called.append(codes)
        return SimpleNamespace(graph=graph(False), project_counterfactual_graph=graph(True),
            project_compilation_evidence=({"kind": "dispatch_fixture"},))

    monkeypatch.setattr(executor_module, "compile_nonplay_complete_graph_v1", compile_pair)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact",
        "visibility": "complete_state", "claim": "winning_root_action_count", "targetTeam": "west_east",
        "utility": "awarded_point_margin", "successThreshold": 1, "budgetProfile": "test"})
    scenario = ResearchScenarioV1.build({"schemaVersion": 1,
        "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        "constraints": _leaf("PRED-PROJECT-PHASE", "state.phase", {"state": "playing"}, "report"),
        "decisionPoint": decision,
        "outcome": _leaf("PRED-PROJECT-GAIN", "tactic.verified", {"kind": "project_bidding",
            "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True}, hardness)})
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-PROJECT-PAIR", bundle_hash="a" * 64,
        tactical_detector_profile=TacticalDetectorProfileV1.defaults()).execute(scenario, request,
            maximum_proof_nodes=1000, maximum_proof_depth=40, maximum_seconds=30)
    assert called and result.status == "qualified", result.execution_receipt
    evidence = result.proof["nativeQualification"]["PRED-PROJECT-GAIN"]
    assert evidence["satisfied"] is True
    assert evidence["observed"]["rootProjectConsequence1"] == 12
    assert result.predicate_evaluation.satisfied is True


@pytest.mark.parametrize("tactic_hardness", [None, "hard", "report"])
def test_native_optional_project_exact_root_preserves_forced_pass(root: Path, tactic_hardness) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    deck = list(range(32))
    for left, right in ((9, 20), (18, 15), (30, 7)):
        deck[left], deck[right] = deck[right], deck[left]

    prefix = _optional_project_prefix_v1(deck_order=deck)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    decision = {"kind": "project", "selector": {"owner": "east", "projectIndex": 0}}
    normalized = normalizer.normalize(prefix, decision_point=decision, maximum_seconds=15)
    assert normalized.next_actor == "east"
    assert [row["code"] for row in normalized.legal_actions] == ["pass"]
    assert normalizer.normalize(prefix, decision_point=decision, maximum_seconds=15).exact_state_id == normalized.exact_state_id
    advanced = dict(prefix)
    advanced["projectActions"] = [*prefix["projectActions"], {"actor": "east", "action": "pass"}]
    after = normalizer.normalize(advanced, decision_point={"kind": "play"}, maximum_seconds=15)
    assert after.next_actor == "east"
    assert after.legal_actions

    proof = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": "west_east",
        "utility": "awarded_point_margin", "successThreshold": -100000,
        "budgetProfile": "test", "horizon": {"unit": "plies", "value": 2},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, bundle_revision_id="CBR-TEST-FORCED-PROJECT", bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    )
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1,
        "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        "constraints": _leaf("PRED-FORCED-CONTRACT", "auction.contract", {"contract": "sun"}),
        "decisionPoint": decision,
        "outcome": _leaf("PRED-FORCED-REPORT", "auction.contract", {"contract": "sun"}, "report"),
    })
    if tactic_hardness is not None:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [raw["constraints"],
            _leaf("PRED-FORCED-ABSENT", "tactic.applicable",
                {"kind": "only_winning_move", "applicable": False}, tactic_hardness)]}
        raw["outcome"] = _leaf("PRED-FORCED-UNVERIFIED", "tactic.verified",
            {"kind": "only_winning_move", "minimumScore": 0, "minimumEvidence": 0}, "report")
        scenario = ResearchScenarioV1.build(raw)
    result = executor.execute(
        scenario, proof, maximum_proof_nodes=1000, maximum_proof_depth=4,
        maximum_seconds=20,
    )
    assert result.status == "qualified", result.execution_receipt
    assert result.proof["completionStatus"] == "bounded_complete"
    assert result.proof["winningRootActionCount"] == 1
    if tactic_hardness is not None:
        native = result.proof["nativeQualification"]
        assert native["PRED-FORCED-ABSENT"]["satisfied"] is True
        assert native["PRED-FORCED-UNVERIFIED"]["satisfied"] is False
        for row in native.values():
            assert row["observed"]["rootActionCount"] == 1
            assert row["observed"]["applicable"] is False
            assert row["observed"]["verified"] is False
            assert row["observed"]["inapplicabilityReason"] == "required_alternative_absent"


@pytest.mark.parametrize("actor_form", ["explicit", "omitted", "mixed", "null"])
@pytest.mark.parametrize("visibility,mode", [
    ("complete_state", "exact"), ("complete_state", "policy_conditioned"),
    ("actor_information_set", "exact"),
])
def test_native_forced_last_card_accepts_tactic_non_applicability(root, monkeypatch, visibility, mode, actor_form):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))
    prefix = fixture["exactTracePrefix"]
    prefix.update(stateAssertions={}, projectAssertions=[])
    prefix = ExactTracePrefixV1.build(prefix)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    for card in fixture["expected"]["principalVariation"][:8]:
        normalized = normalizer.normalize(prefix, decision_point={"kind": "play"})
        prefix = normalizer.append_normalized_action(prefix, normalized, card)
    assert len(prefix.play_actions) == 31
    authored = prefix.to_dict()
    authored.pop("sha256")
    for index, action in enumerate(authored["playActions"]):
        if actor_form == "omitted" or (actor_form == "mixed" and index % 2):
            action.pop("actor", None)
        elif actor_form == "null":
            action["actor"] = None
    prefix = ExactTracePrefixV1.build(authored)
    tactics = ("only_winning_move", "small_card_big_gain")
    if visibility == "complete_state":
        tactics += ("entry_unblocking", "trump_control", "partner_enablement")
    scenario = ResearchScenarioV1.build({"schemaVersion": 1,
        "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        "constraints": _leaf("PRED-LAST-CARD-ABSENT", "tactic.applicable",
            {"kind": "only_winning_move", "applicable": False}),
        "decisionPoint": {"kind": "play", "selector": {"trickIndex": 7, "position": 3}},
        "outcome": {"op": "all", "children": [
            _leaf(f"PRED-LAST-CARD-{tactic.upper().replace('_', '-')}", "tactic.verified",
                {"kind": tactic, "minimumScore": 0, "minimumEvidence": 0}, "report")
            for tactic in tactics]}})
    proof = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": mode,
        "visibility": visibility, "strategyClass": "frozen_policy" if mode == "policy_conditioned" else "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": "west_east",
        "utility": "awarded_point_margin", "successThreshold": -100000,
        "budgetProfile": "late_game_exact_v1",
        **({"policies": _policy_templates()} if mode == "policy_conditioned" else {})})
    materialization = {}
    materialize = executor_module._materialize_play_policy_continuation_v1

    def capture_materialization(**kwargs):
        materialization.update(kwargs)
        return materialize(**kwargs)

    monkeypatch.setattr(executor_module, "_materialize_play_policy_continuation_v1", capture_materialization)
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-FORCED-LAST-CARD", bundle_hash="a" * 64).execute(
            scenario, proof, maximum_proof_nodes=20, maximum_proof_depth=2,
            maximum_hidden_worlds=2, maximum_seconds=30)
    assert result.status == "qualified", result.execution_receipt
    assert result.proof["completionStatus"] == ("policy_complete" if mode == "policy_conditioned" else "exact_complete")
    assert result.proof["winningRootActionCount"] == 1
    assert result.candidate.trace["exactTracePrefix"] == prefix.to_dict()
    assert len(result.candidate.trace["exactTracePrefix"]["playActions"]) == 31
    if visibility == "complete_state":
        complete = result.candidate.trace["completeExactTracePrefix"]
        assert complete["playActions"] == result.candidate.trace["completePlay"]["plays"]
        assert len(complete["playActions"]) == 32
        assert all(row["actor"] is not None for row in complete["playActions"])
        assert result.candidate.trace["completeExactNormalization"]["prefixHash"] == (
            executor_module._effective_prefix_hash(ExactTracePrefixV1.build(complete)))
        if actor_form == "explicit":
            source = materialization["candidate"]
            trace = copy.deepcopy(dict(source.trace))
            wrong = copy.deepcopy(authored)
            wrong["playActions"][0]["actor"] = "west"
            assert authored["playActions"][0]["actor"] != "west"
            trace["exactTracePrefix"] = ExactTracePrefixV1.build(wrong).to_dict()
            with pytest.raises(executor_module._PolicyContinuationCompilerDisagreement,
                               match="not a legal exact continuation") as error:
                materialize(**{**materialization, "candidate": replace(source, trace=trace)})
            assert isinstance(error.value.__cause__, PlayTailSolveStopped)
            assert error.value.__cause__.solver_status == "infeasible"
    evidence = {row["predicateId"]: row for row in result.predicate_evaluation.evidence}
    assert evidence["PRED-LAST-CARD-ABSENT"]["satisfied"] is True
    assert all(not row["satisfied"] for key, row in evidence.items() if key != "PRED-LAST-CARD-ABSENT")
    for row in evidence.values():
        observed = row["observed"]["observed"] if visibility == "complete_state" else row["observed"]
        assert observed["rootActionCount"] == 1
        assert observed["score"] == 0


@pytest.mark.parametrize("mode", ["exact", "bounded"])
@pytest.mark.parametrize("decision_kind", ["project", "play"])
def test_forced_quality_uses_a_constant_size_cp_sat_certificate(mode, decision_kind) -> None:
    action = "pass" if decision_kind == "project" else "S7"
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1,
        "mode": mode,
        "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count",
        "successThreshold": 0,
        "targetTeam": "west_east",
        "utility": "awarded_point_margin",
        "budgetProfile": "bounded_v1",
        **({"horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1"} if mode == "bounded" else {}),
    })
    normalized = SimpleNamespace(
        decision_kind=decision_kind,
        next_actor="west",
        legal_actions=({"code": action},),
        exact_state_id="STATE-FORCED-QUALITY",
    )

    selected, artifact = executor_module._forced_quality_artifact_v1(
        normalized,
        request,
        actual_action=None,
        maximum_seconds=1,
        cancel_check=None,
    )

    quality = artifact["proofModel"]["actionQuality"]
    quality_core = {key: value for key, value in quality.items() if key != "sha256"}
    quality_core["modelProto"] = artifact["proofModel"]["modelProto"]
    assert selected == action
    assert artifact["proof"]["selectedAction"] == action
    assert artifact["proof"]["winningRootActionCount"] is None
    assert artifact["proof"]["winningRootActionLowerBound"] == 0
    assert artifact["proof"]["winningRootActionUpperBound"] == 1
    assert artifact["proof"]["claimSatisfied"] is None
    assert artifact["proof"]["completionScope"] == "action_optimality_only"
    assert artifact["proof"]["rootActions"][0]["value"] is None
    assert artifact["proof"]["rootActions"][0]["satisfies"] is None
    assert quality["bestValue"] is None and quality["actualValue"] is None
    assert quality["actualActionOptimal"] is True
    assert quality["regret"] == 0
    assert quality["sha256"] == canonical_sha256(quality_core)
    assert artifact["proof"]["identities"]["modelHash"] == canonical_sha256(
        artifact["proofModel"]
    )


@pytest.mark.real_clock
@pytest.mark.parametrize("visibility", ("actor_information_set", "complete_state"))
@pytest.mark.parametrize("quality_mode", ("budgeted", "exact"))
def test_forced_project_quality_round_trips_through_evidence_validation(root: Path, monkeypatch, visibility, quality_mode) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    deck = list(range(32))
    for left, right in ((9, 20), (18, 15), (30, 7)):
        deck[left], deck[right] = deck[right], deck[left]
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1(deck_order=deck))
    if visibility == "complete_state":
        def unexpected_graph(*args, **kwargs):
            raise AssertionError("a forced action must not enumerate continuations")
        monkeypatch.setattr(executor_module, "compile_nonplay_complete_graph_v1", unexpected_graph)
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2,
        "visibility": visibility,
        "mode": quality_mode,
        "decisions": {
            kind: {
                "utility": "awarded_point_margin",
                "budgetProfile": "test",
                **({"horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
                    "boundaryEvaluator": "current_raw_point_margin_v1"} if quality_mode == "budgeted" else {}),
            }
            for kind in SCENARIO_DECISION_KINDS
        },
    })
    stopped = False

    def checkpoint(_value):
        nonlocal stopped
        stopped = _value["kind"] == "quality_decision_checkpoint_v1"

    result = resolve_quality_play_continuation_v1(
        rules,
        prefix,
        policy,
        maximum_nodes=1_000,
        maximum_depth=8,
        maximum_worlds=1,
        maximum_seconds=30,
        maximum_decision_seconds=15,
        maximum_evidence_bytes=10_000_000,
        proof_budget_profiles={"test": _quality_budget_profile(root)},
        cancel_check=lambda: stopped,
        checkpoint_callback=checkpoint,
    )

    assert result["status"] == "incomplete"
    assert result["partialReceipt"]["exhaustionReason"] == "cancelled"
    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    assert decision["decisionKind"] == "project"
    assert decision["action"] == "pass"
    assert decision["actualActionOptimal"] is True
    assert result["artifacts"][decision["proofArtifactHash"]]["proofModel"]["kind"] == (
        "forced_action_optimality_model_v1"
    )
    validation = dict(source_prefix=prefix, rules_hash=rules.sha256,
        request=policy.request_for("play", "north"), policy=policy)
    executor_module.validate_native_decision_quality_evidence_v1(result, **validation)
    if visibility == "complete_state":
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
        different = prefix.to_dict()
        different.pop("sha256", None)
        different["matchScore"] = [1, 0]
        with pytest.raises(ValueError, match="owned prefix"):
            executor_module._complete_state_quality_artifact_v1(rules, normalized,
                ExactTracePrefixV1.build(different), {"kind": "project"},
                policy.request_for("project", normalized.next_actor), maximum_nodes=1000,
                maximum_depth=8, maximum_seconds=1, cancel_check=None)
    for count in (True, 1.0, None, -1):
        changed = copy.deepcopy(result)
        row = changed["decisions"][0]
        row["optimalActionCount"] = count
        row["sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "sha256"})
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="optimal action count"):
            executor_module.validate_native_decision_quality_evidence_v1(changed, **validation)
    for section, field, value in (
        ("proof", "winningRootActionCount", 1), ("proof", "claimSatisfied", True),
        ("proof", "winningRootActionLowerBound", False), ("proof", "completionScope", "whole_round"),
        ("quality", "actualValue", 0), ("quality", "bestValue", 0), ("quality", "regret", False),
        ("root", "value", 0), ("root", "satisfies", True),
    ):
        changed = copy.deepcopy(result)
        row = changed["decisions"][0]
        artifact = changed["artifacts"].pop(row["proofArtifactHash"])
        destination = (artifact["proof"] if section == "proof" else
            artifact["proofModel"]["actionQuality"] if section == "quality" else artifact["proof"]["rootActions"][0])
        destination[field] = value
        digest = canonical_sha256(artifact)
        changed["artifacts"][digest] = artifact
        row["proofArtifactHash"] = digest
        row["sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "sha256"})
        changed["sha256"] = canonical_sha256({k: v for k, v in changed.items() if k not in {"sha256", "executionReceipt"}})
        with pytest.raises(ValueError, match="forced optimality cannot claim"):
            executor_module.validate_native_decision_quality_evidence_v1(changed, **validation)

    # Historical receipts retain their old interpretation and immutable hashes.
    legacy = copy.deepcopy(result)
    row = legacy["decisions"][0]
    artifact = legacy["artifacts"].pop(row["proofArtifactHash"])
    model, proof = artifact["proofModel"], artifact["proof"]
    model["kind"] = "forced_legal_action_model_v1"
    quality = model["actionQuality"]
    quality.update(bestValue=0, actualValue=0)
    quality["sha256"] = canonical_sha256({**{k: v for k, v in quality.items() if k != "sha256"},
        "modelProto": model["modelProto"]})
    proof.update(claimSatisfied=True, winningRootActionCount=1)
    for key in ("completionScope", "winningRootActionLowerBound", "winningRootActionUpperBound"):
        proof.pop(key)
    proof["rootActions"][0].update(value=0, satisfies=True)
    proof["identities"]["modelHash"] = canonical_sha256(model)
    proof["sha256"] = canonical_sha256({k: v for k, v in proof.items() if k != "sha256"})
    digest = canonical_sha256(artifact)
    legacy["artifacts"][digest] = artifact
    row.update(proofArtifactHash=digest, actionQualityHash=quality["sha256"])
    row["sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "sha256"})
    legacy["sha256"] = canonical_sha256({k: v for k, v in legacy.items() if k not in {"sha256", "executionReceipt"}})
    executor_module.validate_native_decision_quality_evidence_v1(legacy, **validation)


def test_optional_project_search_selection_rejects_forced_pass(root: Path) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    decision_point = {
        "kind": "project",
        "selector": {"owner": "east", "projectIndex": 0},
    }
    source = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        _optional_project_prefix_v1(),
        decision_point=decision_point,
        maximum_seconds=15,
    )
    receipt = source.normalization_receipt.to_dict()
    projects = copy.deepcopy(receipt["derivedProjects"])
    projects["declarationActions"].append({"actor": "east", "action": "pass"})
    for row in projects["projects"]:
        if row["owner"] == "east":
            row["selected"] = False
            row["declared"] = False
    semantic_hash = canonical_sha256({"kind": "forced-project-pass-fixture"})
    candidate = OrToolsNeutralCandidateV1(
        "search",
        {
            "deckOrder": list(range(32)),
            "dealer": "east",
            "roundNumber": 1,
            "matchScore": [0, 0],
            "auction": {
                "completion": {
                    "actions": [{"actor": "north", "action": "sun"}]
                }
            },
            "escalation": receipt["derivedEscalation"],
            "projects": projects,
            "play": receipt["derivedPlay"],
        },
        {},
        tuple(range(32)),
        {
            "fixture": OrToolsResearchStageReceiptV1(
                "fixture",
                semantic_hash,
                semantic_hash,
                semantic_hash,
                "optimal",
            )
        },
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-FORCED-PROJECT-ROOT",
                "state.phase",
                {"state": "playing"},
                "report",
            ),
            "decisionPoint": decision_point,
            "outcome": _leaf(
                "PRED-FORCED-PROJECT-REPORT",
                "state.phase",
                {"state": "playing"},
                "report",
            ),
        }
    )

    with pytest.raises(
        ProjectDecisionNotStrategicError,
        match="project_decision_not_strategic",
    ):
        executor_module._search_decision_prefix(
            candidate,
            scenario,
            project_declaration_mode="optional",
        )
    native_prefix = executor_module._search_decision_prefix(
        candidate, scenario, project_declaration_mode="optional", native_cp_sat=True,
    )
    # Prefix extraction is syntax, not a claim that the supplied project
    # evidence is true. Native normalization must derive the actual legal set.
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        native_prefix, decision_point=decision_point, maximum_seconds=15,
    )
    assert [row["code"].split(":", 1)[0] for row in normalized.legal_actions] == ["pass", "declare"]


def test_final_project_native_pointwise_dominance_and_fresh_verification(root):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    from baloot_platform.research.scenario_contracts import canonical_sha256

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "project", "selector": {"owner": "east", "projectIndex": 0}})
    actions = sorted(row["code"] for row in exact.legal_actions)
    assert len(actions) == 2 and actions[1] == "pass"
    raw = dict(schemaVersion=1, mode="exact", visibility="complete_state",
        strategyClass="pure_deterministic", claim="outcome_forced", targetTeam="west_east",
        utility="awarded_point_margin", successThreshold=0, budgetProfile="late_game_exact_v1",
        selectedAction=actions[0])
    request = ScenarioProofRequestV1.build(raw)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    original = exact.to_dict()
    budget_raw = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    budget = ScenarioProofBudgetV1.build(request.budget_profile, budget_raw)
    limits = dict(maximum_seconds=60, budget=budget, maximum_conflicts=100_000)
    proof = compiler._prove_last_project_dominance(exact, prefix, request, **limits)
    assert proof["status"] == "strictly_dominant"
    assert proof["rootValues"] is None and proof["winningRootActionCount"] is None
    assert not proof["wholeRoundCertified"]
    assert [row["solverStatus"] for row in proof["queries"]] == ["optimal", "infeasible"]
    assert compiler._verify_last_project_dominance(proof, exact, prefix, request,
        **limits) == proof
    forged = copy.deepcopy(proof)
    forged["queries"][0]["selectedMargin"] += 1
    forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "sha256"})
    with pytest.raises(ValueError, match="fresh native proof"):
        compiler._verify_last_project_dominance(forged, exact, prefix, request, **limits)
    reversed_request = ScenarioProofRequestV1.build({**raw, "selectedAction": "pass"})
    reverse = compiler._prove_last_project_dominance(exact, prefix, reversed_request, **limits)
    assert reverse["status"] == "counterexample_found"
    assert reverse["queries"][-1]["selectedMargin"] <= reverse["queries"][-1]["alternativeMargin"]
    assert exact.to_dict() == original
    with pytest.raises(ValueError, match="owned exact final Sun"):
        compiler._prove_last_project_dominance(exact, prefix,
            ScenarioProofRequestV1.build({**raw, "visibility": "actor_information_set"}), maximum_seconds=20)
    from baloot_platform.research.result_authority import (
        build_native_project_dominance_quality_artifact_v1,
        validate_native_project_dominance_quality_artifact_v1,
    )
    context = dict(rules=rules, source_prefix=prefix, normalized=exact, request=request, **limits)
    artifact = build_native_project_dominance_quality_artifact_v1(proof, **context)
    quality = artifact["proofModel"]["actionQuality"]
    assert quality["actualActionOptimal"] and quality["optimalActionCount"] == 1
    assert quality["bestValue"] is None and quality["actualValue"] is None
    assert artifact["proof"]["claimSatisfied"] is None
    assert artifact["proof"]["winningRootActionCount"] is None
    validate_native_project_dominance_quality_artifact_v1(artifact, **context)
    forged_artifact = copy.deepcopy(artifact)
    forged_artifact["proof"]["claimSatisfied"] = True
    forged_artifact["proof"]["sha256"] = canonical_sha256({k: v for k, v in forged_artifact["proof"].items() if k != "sha256"})
    with pytest.raises(ValueError, match="artifact binding"):
        validate_native_project_dominance_quality_artifact_v1(forged_artifact, **context)
    with pytest.raises(ValueError, match="strictly dominant"):
        build_native_project_dominance_quality_artifact_v1(reverse, **{**context, "request": reversed_request})
    unselected = ScenarioProofRequestV1.build({k: v for k, v in raw.items() if k != "selectedAction"})
    with pytest.raises(ValueError, match="strictly dominant"):
        build_native_project_dominance_quality_artifact_v1(proof, **{**context, "request": unselected})
    generated = compiler._select_last_project_dominance(exact, prefix, unselected, **limits)
    assert generated["status"] == "strictly_dominant"
    assert generated["request"] == unselected.to_dict()
    assert generated["selectedAction"] == request.selected_action
    assert generated["queryCount"] == 2
    assert generated["attempts"][0]["request"] == request.to_dict()
    assert generated["attempts"][0]["proof"] == proof
    generated_context = {**context, "request": unselected}
    generated_artifact = build_native_project_dominance_quality_artifact_v1(generated, **generated_context)
    assert generated_artifact["request"] == unselected.to_dict()
    assert generated_artifact["proof"]["selectedAction"] == request.selected_action
    assert generated_artifact["proof"]["claimSatisfied"] is None
    for field in ("request", "legalActions"):
        altered = copy.deepcopy(generated)
        if field == "request":
            altered["attempts"][0]["request"]["selectedAction"] = "pass"
        else:
            altered["legalActions"].reverse()
        altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="fresh native proof"):
            build_native_project_dominance_quality_artifact_v1(altered, **generated_context)
    changed_budget = ScenarioProofBudgetV1.build(request.budget_profile, {**budget_raw, "maximumDepth": 65})
    with pytest.raises(ValueError, match="frozen proof budget"):
        build_native_project_dominance_quality_artifact_v1(proof, **{**context, "budget": changed_budget})


@pytest.mark.real_clock
@pytest.mark.parametrize("audited", [True, False])
def test_observed_project_quality_admission_preserves_negative_audits(root, monkeypatch, audited):
    """Orchestration only; native bounds/reconstruction have separate real tests."""
    from baloot_platform.research import result_authority as authority
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exact = normalizer.normalize_play_or_project(prefix)
    declared = next(row["code"] for row in exact.legal_actions if row["code"] != "pass")
    recorded = normalizer.append_normalized_action(prefix, exact, "pass")
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: {"utility": "awarded_point_margin",
            "budgetProfile": "test"} for kind in SCENARIO_DECISION_KINDS}})
    profile = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    verified = []
    monkeypatch.setattr(authority, "reverify_native_observed_project_bounds_v1",
        lambda *args, **kwargs: verified.append(kwargs["selected_root_action"]))

    def artifact(received_rules, normalized, current, request, **kwargs):
        assert kwargs["decision_kind"] == "project" and kwargs["escalation_events"] == []
        action = request.selected_action or declared
        limits = {"iterations": 1, "queries": 5, "observations": 1, "modelBytes": 1,
            "conflictsPerSolve": 1000, "depth": 32, "seconds": 1}
        selected = {"selectedRootAction": action, "queriesUsed": 1, "limits": limits,
            "strategy": {"lowerBound": 0 if action == "pass" else 1}, "sha256": "a" * 64}
        best = {"queriesUsed": 1, "limits": limits,
            "strategy": {"upperBound": 1}, "sha256": "b" * 64}
        return authority.build_native_observed_play_quality_artifact_v1(selected, best,
            rules=received_rules, source_prefix=current, normalized=normalized, request=request,
            budget=kwargs["budget"], opportunity_order=kwargs["opportunity_order"], escalation_events=[],
            maximum_conflicts=1000, maximum_seconds=10, decision_kind="project")

    monkeypatch.setattr(executor_module, "_observed_play_quality_artifact_v1", artifact)
    checkpoints = []
    def checkpoint(row):
        if row.get("kind") == "quality_decision_checkpoint_v1":
            assert len(verified) == 2  # Native replay precedes early-stop/cache consumers.
        checkpoints.append(row)
    result = resolve_quality_play_continuation_v1(rules, prefix, policy,
        **({"recorded_trace": recorded, "continue_recorded_prefix": True} if audited else {}),
        stop_on_recorded_quality_failure=True, checkpoint_callback=checkpoint,
        cancel_check=lambda: any(row.get("kind") == "quality_decision_checkpoint_v1" for row in checkpoints),
        maximum_nodes=100_000, maximum_depth=64, maximum_worlds=1, maximum_seconds=120,
        maximum_decision_seconds=60, maximum_evidence_bytes=30_000_000,
        opportunity_order=("west", "east", "north", "south"), escalation_events=[],
        proof_budget_profiles={"test": profile}, search_budget_profiles={"test": {"maximumConflicts": 1000}})
    assert result["status"] == ("quality_failed" if audited else "incomplete"), result.get("partialReceipt")
    if audited:
        assert result["partialReceipt"]["evidence"]["generationReceipt"]["reason"] == "recorded_action_not_optimal"
        assert result["partialReceipt"]["evidence"]["readbackReceipt"]["exhaustionReason"] == "cancelled"
    assert len(result["decisions"]) == 1
    row = result["decisions"][0]
    assert row["decisionKind"] == "project" and row["actualActionOptimal"] is not audited
    assert row["regret"] == int(audited) and row["optimalActionCount"] is None
    assert row["action"] == ("pass" if audited else declared)
    assert len(verified) == 2  # No duplicate immediate-final replay.
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, rules=rules,
        request=policy.request_for("play", "north"), policy=policy,
        search_budget_profiles={"test": {"maximumConflicts": 1000}},
        opportunity_order=("west", "east", "north", "south"), escalation_events=[])
    authority.validate_native_decision_quality_evidence_v1(result, **context)
    assert len(verified) == 4  # A distinct read always replays again.
    owned = frozenset({row["proofArtifactHash"]})
    with pytest.raises(TypeError, match="verified_artifacts"):
        authority.validate_native_decision_quality_evidence_v1(result, **context, verified_artifacts=owned)
    changed = copy.deepcopy(result)
    changed["decisions"][0]["regret"] += 1
    changed["decisions"][0]["sha256"] = canonical_sha256({key: value for key, value in changed["decisions"][0].items() if key != "sha256"})
    changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="decision binding mismatch"):
        authority._validate_native_decision_quality_evidence_v1(changed, **context, verified_artifacts=owned)
    assert len(verified) == 4
    changed = copy.deepcopy(result)
    altered = changed["artifacts"].pop(row["proofArtifactHash"])
    altered["proof"]["winningRootActionCount"] = 1
    altered["proof"]["sha256"] = canonical_sha256({key: value for key, value in altered["proof"].items() if key != "sha256"})
    digest = canonical_sha256(altered)
    changed["artifacts"][digest] = altered
    changed["decisions"][0]["proofArtifactHash"] = digest
    changed["decisions"][0]["sha256"] = canonical_sha256({key: value for key, value in changed["decisions"][0].items() if key != "sha256"})
    changed["sha256"] = canonical_sha256({key: value for key, value in changed.items() if key not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="artifact binding mismatch"):
        authority._validate_native_decision_quality_evidence_v1(changed, **context, verified_artifacts=owned)
    assert len(verified) == 6

    import time
    for stop in (None, "cancelled", "maximum_seconds"):
        clock, cancelled, calls = [0.0], [False], []
        def cancel():
            return cancelled[0]
        def normalize(self, supplied, **kwargs):
            assert supplied.sha256 == prefix.sha256 and kwargs["maximum_seconds"] == 10
            assert kwargs["cancel_check"]() is False
            calls.append("normalization")
            clock[0] = 10 if stop == "maximum_seconds" else 3
            cancelled[0] = stop == "cancelled"
            return exact
        def validate(supplied, **kwargs):
            assert supplied == result["artifacts"][row["proofArtifactHash"]]
            assert kwargs["maximum_seconds"] == 7
            assert kwargs["cancel_check"]() is False
            cancelled[0] = True
            assert kwargs["cancel_check"]() is True
            calls.append("proof")
            raise RuntimeError("observed budget reached native proof")
        with monkeypatch.context() as nested:
            nested.setattr(time, "monotonic", lambda: clock[0])
            nested.setattr(OrToolsExactStateNormalizerV1, "normalize", normalize)
            nested.setattr(authority, "validate_native_observed_play_quality_artifact_v1", validate)
            if stop is None:
                with pytest.raises(RuntimeError, match="observed budget reached native proof"):
                    authority.validate_native_decision_quality_evidence_v1(result, **context,
                        maximum_seconds=10, cancel_check=cancel)
                assert calls == ["normalization", "proof"]
            else:
                with pytest.raises(executor_module.InformationSetCompilationStopped) as caught:
                    authority.validate_native_decision_quality_evidence_v1(result, **context,
                        maximum_seconds=10, cancel_check=cancel)
                assert caught.value.reason == stop
                assert calls == ["normalization"]


@pytest.mark.parametrize("decision_count,resume_after_opening", [
    pytest.param(1, False, id="1"), pytest.param(2, False, id="2"),
    pytest.param(1, True, id="west-project"),
])
def test_observed_optional_project_production_checkpoint(root, tmp_path, record_property, decision_count, resume_after_opening):
    """Real non-forced declaration admission; cancellation is not round completion."""
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    deck = [None] * 32
    for seat in Seat:
        for position, card in zip(layout.positions_for(seat, final=True),
                range(int(seat) * 8, int(seat) * 8 + 8), strict=True):
            deck[position] = card
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, deckOrder=deck, dealer="east",
        auctionActions=[{"actor": "north", "action": "sun"}], escalationActions=[],
        projectDeclarations=[], projectActions=[], playActions=[]))
    if resume_after_opening:
        # Reproduce the retained checkpoint's exact input, not its prior proof.
        payload = prefix.to_dict()
        payload.pop("sha256")
        payload["projectActions"] = [{"actor": "north", "action": "declare",
            "projectId": "PROJECT-7CA4C38B00EABA58"}]
        payload["playActions"] = [{"actor": "north", "card": 3, "ekka": False}]
        prefix = ExactTracePrefixV1.build(payload)
        assert prefix.sha256 == "fff215af5e070386fb730208351f627346c61488522bf7694c3cdf4ffce008af"
    profiles = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]
    profile_id = "offline_heavyweight_v1"
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "actor_information_set", "decisions": {kind: {"utility": "awarded_point_margin",
            "budgetProfile": profile_id} for kind in SCENARIO_DECISION_KINDS}})
    search = {profile_id: {"maximumConflicts": 250_000}}
    order = ("west", "east", "north", "south")
    checkpoints = []
    result = resolve_quality_play_continuation_v1(rules, prefix, policy,
        checkpoint_callback=checkpoints.append,
        cancel_check=lambda: sum(row.get("kind") == "quality_decision_checkpoint_v1" for row in checkpoints) >= decision_count,
        maximum_nodes=100_000, maximum_depth=64, maximum_worlds=1, maximum_seconds=180 * decision_count,
        maximum_decision_seconds=180, maximum_evidence_bytes=30_000_000,
        opportunity_order=order, escalation_events=[],
        proof_budget_profiles=profiles, search_budget_profiles=search)
    receipt = tmp_path / "observed-optional-project-production.json"
    receipt.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    record_property("observedOptionalProjectProductionEvidencePath", str(receipt))
    assert result["status"] == "incomplete", result.get("partialReceipt")
    assert result["partialReceipt"]["exhaustionReason"] == "cancelled", result.get("partialReceipt")
    assert len(result["decisions"]) == decision_count
    row = result["decisions"][0]
    assert row["decisionKind"] == "project"
    assert row["actor"] == ("west" if resume_after_opening else "north")
    assert row["sourcePrefixHash"] == prefix.sha256
    if not resume_after_opening:
        assert row["action"].startswith("declare:")
    assert row["actualActionOptimal"] is True and row["regret"] == 0
    assert row["optimalActionCount"] is None
    artifact = result["artifacts"][row["proofArtifactHash"]]
    assert artifact["proofModel"]["kind"] == "native_observed_project_quality_model_v1"
    assert artifact["graph"]["actualValue"] == artifact["graph"]["bestValue"]
    if not resume_after_opening:
        assert artifact["graph"]["bestValue"] == 68
    if decision_count == 2:
        play = result["decisions"][1]
        assert play["decisionKind"] == "play" and play["actualActionOptimal"] is True
        assert play["regret"] == 0 and play["optimalActionCount"] is None
        play_artifact = result["artifacts"][play["proofArtifactHash"]]
        assert play_artifact["graph"]["actualValue"] == play_artifact["graph"]["bestValue"] == 68
        assert play_artifact["graph"]["bestProof"]["preparation"]["observedPlayBoundsVersion"] == 2
    executor_module.validate_native_decision_quality_evidence_v1(result,
        source_prefix=prefix, rules_hash=rules.sha256, rules=rules,
        request=policy.request_for("play", "north"), policy=policy,
        search_budget_profiles=search, opportunity_order=order, escalation_events=[])


@pytest.mark.real_clock
@pytest.mark.parametrize("audited", [True, False])
def test_final_project_dominance_audit_route_and_frozen_readback(root, monkeypatch, audited):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exact = normalizer.normalize_play_or_project(prefix)
    selected = next(row["code"] for row in exact.legal_actions if row["code"] != "pass")
    recorded = normalizer.append_normalized_action(prefix, exact, selected)
    policy = ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2, "mode": "exact",
        "visibility": "complete_state", "decisions": {kind: {"utility": "awarded_point_margin",
            "budgetProfile": "test"} for kind in SCENARIO_DECISION_KINDS}})
    budget = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    search = {"test": {"maximumConflicts": 100_000}}
    monkeypatch.setattr(executor_module, "compile_nonplay_complete_graph_v1",
        lambda *args, **kwargs: pytest.fail("final project audit entered exhaustive graph fallback"))
    checkpoints = []
    result = resolve_quality_play_continuation_v1(rules, prefix, policy,
        **({"recorded_trace": recorded, "continue_recorded_prefix": True} if audited else {}),
        checkpoint_callback=checkpoints.append,
        cancel_check=lambda: any(row.get("kind") == "quality_decision_checkpoint_v1" for row in checkpoints),
        maximum_nodes=100_000, maximum_depth=64, maximum_worlds=1, maximum_seconds=120,
        maximum_decision_seconds=60, maximum_evidence_bytes=30_000_000,
        proof_budget_profiles={"test": budget}, search_budget_profiles=search)
    assert result["status"] == "incomplete", result.get("partialReceipt")
    assert result["partialReceipt"]["exhaustionReason"] == "cancelled", result["partialReceipt"]
    assert len(result["decisions"]) == 1
    row = result["decisions"][0]
    assert row["action"] == selected and row["actualActionOptimal"] is True
    assert row["optimalActionCount"] == 1
    artifact = result["artifacts"][row["proofArtifactHash"]]
    assert artifact["graph"]["kind"] == ("native_last_project_dominance_v1" if audited else "native_last_project_selection_v1")
    assert artifact["proofModel"]["kind"] == "native_project_dominance_quality_model_v1"
    assert artifact["proof"]["claimSatisfied"] is None
    context = dict(source_prefix=prefix, rules_hash=rules.sha256, rules=rules,
        request=policy.request_for("play", "north"), policy=policy)
    executor_module.validate_native_decision_quality_evidence_v1(result, **context, search_budget_profiles=search)
    narrowed = copy.deepcopy(result)
    narrowed["limits"]["maximumDepth"] = 1
    narrowed["sha256"] = canonical_sha256({k: v for k, v in narrowed.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="frozen proof budget"):
        executor_module.validate_native_decision_quality_evidence_v1(narrowed, **context, search_budget_profiles=search)
    with pytest.raises(ValueError, match="frozen search limits"):
        executor_module.validate_native_decision_quality_evidence_v1(result, **context)
    with pytest.raises(ValueError, match="frozen proof budget"):
        executor_module.validate_native_decision_quality_evidence_v1(result, **context,
            search_budget_profiles={"test": {"maximumConflicts": 100_001}})
    forged = copy.deepcopy(result)
    altered = forged["artifacts"].pop(row["proofArtifactHash"])
    altered["proof"]["claimSatisfied"] = True
    altered["proof"]["sha256"] = canonical_sha256({k: v for k, v in altered["proof"].items() if k != "sha256"})
    digest = canonical_sha256(altered)
    forged["artifacts"][digest] = altered
    forged["decisions"][0]["proofArtifactHash"] = digest
    forged["decisions"][0]["sha256"] = canonical_sha256({k: v for k, v in forged["decisions"][0].items() if k != "sha256"})
    forged["sha256"] = canonical_sha256({k: v for k, v in forged.items() if k not in {"sha256", "executionReceipt"}})
    with pytest.raises(ValueError, match="artifact binding"):
        executor_module.validate_native_decision_quality_evidence_v1(forged, **context, search_budget_profiles=search)

    # The nested normalization/proof share the caller's allowance, not a fresh tier.
    import time
    from baloot_platform.research import result_authority as authority
    for stop in (None, "cancelled", "maximum_seconds"):
        clock, cancelled, calls = [0.0], [False], []
        def cancel():
            return cancelled[0]
        def normalize(self, supplied, **kwargs):
            assert supplied.sha256 == prefix.sha256
            assert kwargs["maximum_seconds"] == 10 and kwargs["cancel_check"] is cancel
            calls.append("normalization")
            clock[0] = 10 if stop == "maximum_seconds" else 3
            cancelled[0] = stop == "cancelled"
            return exact
        def validate(supplied, **kwargs):
            assert supplied == artifact
            assert kwargs["maximum_seconds"] == 7 and kwargs["cancel_check"] is cancel
            calls.append("proof")
            raise RuntimeError("project budget reached native proof")
        with monkeypatch.context() as nested:
            nested.setattr(time, "monotonic", lambda: clock[0])
            nested.setattr(OrToolsExactStateNormalizerV1, "normalize_play_or_project", normalize)
            nested.setattr(authority, "validate_native_project_dominance_quality_artifact_v1", validate)
            if stop is None:
                with pytest.raises(RuntimeError, match="project budget reached native proof"):
                    authority.validate_native_decision_quality_evidence_v1(result, **context,
                        search_budget_profiles=search, maximum_seconds=10, cancel_check=cancel)
                assert calls == ["normalization", "proof"]
            else:
                with pytest.raises(executor_module.InformationSetCompilationStopped) as caught:
                    authority.validate_native_decision_quality_evidence_v1(result, **context,
                        search_budget_profiles=search, maximum_seconds=10, cancel_check=cancel)
                assert caught.value.reason == stop
                assert calls == ["normalization"]


@pytest.mark.real_clock
@pytest.mark.parametrize("statuses", [
    ("strictly_dominant",), ("counterexample_found", "strictly_dominant"),
    ("counterexample_found", "counterexample_found"), ("counterexample_found", "unknown"),
    ("counterexample_found", "normalization_stop"), ("counterexample_found", "path_stop"),
])
def test_generated_project_selector_accounts_for_every_attempt(root, monkeypatch, statuses):
    import baloot_platform.research.ortools_information_set_compiler as module
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        claim="outcome_forced", targetTeam="west_east", utility="awarded_point_margin",
        successThreshold=0, budgetProfile="test"))
    raw_budget = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 4,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    budget = ScenarioProofBudgetV1.build("test", raw_budget)
    compiler = module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    calls = []
    def prove(_exact, _prefix, candidate, **kwargs):
        calls.append(candidate)
        status = statuses[len(calls) - 1]
        if status == "unknown":
            raise module.InformationSetCompilationStopped("unknown", stage="test",
                evidence={"queries": [{"modelBytes": 30}]})
        if status == "normalization_stop":
            raise module.ExactNormalizationStopped("maximum_seconds")
        if status == "path_stop":
            raise module.PlayTailSolveStopped("maximum_seconds")
        return {"status": status, "queries": [{"modelBytes": 10}, {"modelBytes": 20}],
            "sha256": str(len(calls)) * 64, "limits": {"budgetHash": budget.sha256}}
    monkeypatch.setattr(compiler, "_prove_last_project_dominance", prove)
    if statuses[-1] in {"unknown", "normalization_stop", "path_stop"}:
        with pytest.raises(module.InformationSetCompilationStopped, match="unknown|maximum_seconds") as caught:
            compiler._select_last_project_dominance(exact, prefix, request, budget=budget,
                maximum_seconds=20, maximum_conflicts=100_000)
        assert "causeReceipt" in str(caught.value.partial_receipt)
        assert caught.value.partial_receipt["evidence"]["queryCount"] == (3 if statuses[-1] == "unknown" else 2)
        assert caught.value.partial_receipt["evidence"]["peakModelBytes"] == (30 if statuses[-1] == "unknown" else 20)
    else:
        result = compiler._select_last_project_dominance(exact, prefix, request, budget=budget,
            maximum_seconds=20, maximum_conflicts=100_000)
        assert result["queryCount"] == 2 * len(statuses)
        assert result["peakModelBytes"] == 20
        assert result["status"] == ("strictly_dominant" if statuses[-1] == "strictly_dominant" else "no_strictly_dominant_action")
        assert result["selectedAction"] == (calls[-1].selected_action if statuses[-1] == "strictly_dominant" else None)
    assert [candidate.selected_action for candidate in calls] == sorted(row["code"] for row in exact.legal_actions)[:len(statuses)]
    for candidate in calls:
        original = candidate.to_dict()
        original.pop("sha256")
        original.pop("selectedAction")
        assert ScenarioProofRequestV1.build(original) == request
    raw_budget["nativeObserved"]["maximumQueries"] = 3
    with pytest.raises(module.InformationSetCompilationStopped, match="maximum_queries"):
        compiler._select_last_project_dominance(exact, prefix, request,
            budget=ScenarioProofBudgetV1.build("test", raw_budget), maximum_seconds=20, maximum_conflicts=100_000)
    if statuses == ("counterexample_found", "counterexample_found"):
        monkeypatch.setattr(module.OrToolsActorInformationSetCompilerV1, "_select_last_project_dominance",
            lambda *args, **kwargs: result)
        def fallback(*args, **kwargs):
            assert 0 < kwargs["maximum_seconds"] <= 20
            raise executor_module.NonPlayGraphCompilationStoppedV1("test_fallback", nodes=1, depth=1)
        monkeypatch.setattr(executor_module, "compile_nonplay_complete_graph_v1", fallback)
        with pytest.raises(module.InformationSetCompilationStopped, match="test_fallback") as caught:
            executor_module._complete_state_quality_artifact_v1(rules, exact, prefix,
                {"kind": "project"}, request, maximum_nodes=100, maximum_depth=64,
                maximum_seconds=20, cancel_check=None, budget=budget, maximum_conflicts=100_000)
        assert caught.value.partial_receipt["evidence"]["projectDominanceAttempt"] == result


@pytest.mark.parametrize("change", ["hokum", "earlier_declarer"])
def test_final_project_dominance_rejects_unsupported_root(root, change):
    from baloot_platform.research.ortools_information_set_compiler import OrToolsActorInformationSetCompilerV1
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = _optional_project_prefix_v1()
    if change == "hokum":
        raw["auctionActions"] = [{"actor": "north", "action": "hokum"},
            *({"actor": actor, "action": "pass"} for actor in ("west", "south", "east"))]
    else:
        raw["projectActions"] = raw["projectActions"][:2]
        raw["playActions"] = raw["playActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "project", "selector": {"owner": "east" if change == "hokum" else "south"}})
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        strategyClass="pure_deterministic", claim="outcome_forced",
        targetTeam=Seat.parse(exact.next_actor).team.name.lower(), utility="awarded_point_margin",
        successThreshold=0, budgetProfile="late_game_exact_v1", selectedAction=exact.legal_actions[0]["code"]))
    with pytest.raises(ValueError, match="owned exact final Sun"):
        OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prove_last_project_dominance(
            exact, prefix, request, maximum_seconds=20)


def test_earlier_project_declaration_changes_later_legal_action_tree(root):
    """A common play prefix is insufficient for earlier-root pointwise proof."""
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    raw = _optional_project_prefix_v1()
    raw.update(projectActions=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    north = normalizer.normalize_play_or_project(prefix)
    assert north.next_actor == "north"
    assert {row["action"] for row in north.legal_actions} == {"pass", "declare"}
    branches = {}
    for action in north.legal_actions:
        child = normalizer.append_normalized_action(prefix, north, action["code"]).to_dict()
        child.pop("sha256")
        child["playActions"] = _optional_project_prefix_v1()["playActions"][:1]
        west = normalizer.normalize_play_or_project(ExactTracePrefixV1.build(child))
        assert west.next_actor == "west"
        branches[action["action"]] = {row["action"] for row in west.legal_actions}
    assert branches == {"pass": {"pass", "declare"}, "declare": {"pass"}}


@pytest.mark.parametrize("limit", ["maximum_queries", "maximum_depth", "maximum_caller_depth", "maximum_model_bytes", "maximum_cpu_seconds"])
def test_final_project_dominance_enforces_frozen_limits(root, monkeypatch, limit):
    import baloot_platform.research.ortools_information_set_compiler as module
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "project", "selector": {"owner": "east"}})
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        strategyClass="pure_deterministic", claim="outcome_forced", targetTeam="west_east",
        utility="awarded_point_margin", successThreshold=0, budgetProfile="late_game_exact_v1",
        selectedAction=next(row["code"] for row in exact.legal_actions if row["code"] != "pass")))
    raw = _quality_budget_profile(root, maximumDepth=64, maximumHorizonActions=64,
        nativeObserved={"schemaVersion": 1, "maximumIterations": 1000, "maximumQueries": 2000,
            "maximumObservations": 2000, "maximumBoundPrefixes": 2000,
            "maximumCachedPrefixesPerBranch": 2000})
    if limit == "maximum_queries":
        raw["nativeObserved"]["maximumQueries"] = 1
    elif limit == "maximum_depth":
        raw["maximumDepth"] = 1
    elif limit == "maximum_model_bytes":
        raw["maximumCacheBytes"] = 1
    elif limit == "maximum_cpu_seconds":
        raw["maximumCpuSeconds"] = 1
        readings = iter((0.0, 2.0))
        monkeypatch.setattr(module.time, "process_time", lambda: next(readings, 2.0))
    budget = ScenarioProofBudgetV1.build(request.budget_profile, raw)
    with pytest.raises(module.InformationSetCompilationStopped, match=limit.replace("caller_", "")):
        module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prove_last_project_dominance(
            exact, prefix, request, maximum_seconds=20, budget=budget, maximum_conflicts=100_000,
            maximum_depth=1 if limit == "maximum_caller_depth" else 64)


@pytest.mark.parametrize("failure_phase, solver_status", [(0, cp_model.INFEASIBLE), (0, cp_model.UNKNOWN), (1, cp_model.UNKNOWN)])
def test_final_project_dominance_rejects_vacuity_and_incomplete(root, monkeypatch, failure_phase, solver_status):
    import baloot_platform.research.ortools_information_set_compiler as module

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    prefix = ExactTracePrefixV1.build(_optional_project_prefix_v1())
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix,
        decision_point={"kind": "project", "selector": {"owner": "east", "projectIndex": 0}})
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
        strategyClass="pure_deterministic", claim="outcome_forced", targetTeam="west_east",
        utility="awarded_point_margin", successThreshold=0, budgetProfile="late_game_exact_v1",
        selectedAction=next(row["code"] for row in exact.legal_actions if row["code"] != "pass")))
    compiler = module.OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    with pytest.raises(module.InformationSetCompilationStopped, match="cancelled"):
        compiler._prove_last_project_dominance(exact, prefix, request,
            maximum_seconds=20, cancel_check=lambda: True)
    original = module._solve_cp_sat_with_cancel_v1
    calls = []
    def fail_selected_query(*args):
        calls.append(None)
        return solver_status if len(calls) - 1 == failure_phase else original(*args)
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", fail_selected_query)
    exception = ValueError if solver_status == cp_model.INFEASIBLE else module.InformationSetCompilationStopped
    with pytest.raises(exception, match="infeasible|unknown"):
        compiler._prove_last_project_dominance(exact, prefix, request, maximum_seconds=20)
    assert len(calls) == failure_phase + 1


@pytest.mark.parametrize("expected_winners", (2, 0))
def test_optional_project_exact_bounded_proof_uses_pass_and_declare_root(
    root: Path,
    expected_winners: int,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    decision_point = {
        "kind": "project",
        "selector": {"owner": "east", "projectIndex": 0},
    }
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": _optional_project_prefix_v1(),
            },
                "constraints": {"op": "all", "children": [_leaf(
                    "PRED-OPTIONAL-PROJECT-LEGAL-ACTIONS",
                    "auction.contract",
                    {"contract": "sun"},
                ), _leaf("PRED-OPTIONAL-PROJECT-ROOT-COUNT",
                         "proof.winning_root_action_count",
                         {"minimum": expected_winners, "maximum": expected_winners})]},
            "decisionPoint": decision_point,
                "outcome": _leaf(
                    "PRED-OPTIONAL-PROJECT-REPORT",
                    "auction.contract",
                    {"contract": "sun"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "horizon": {"unit": "decisions", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V4-OPTIONAL-PROJECT",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    ).execute(
        scenario,
        proof,
        maximum_proof_nodes=100,
        maximum_proof_depth=2,
        maximum_seconds=20,
    )

    assert result.status == ("qualified" if expected_winners == 2 else "semantic_postcondition_miss"), (
        result.proof,
        result.execution_receipt.get("incompleteCompilation"),
    )
    assert result.proof["completionStatus"] == "bounded_complete"
    assert result.proof["winningRootActionCount"] == 2
    assert {row["action"].split(":", 1)[0] for row in result.proof["rootActions"]} == {
        "pass",
        "declare",
    }


def test_optional_project_policy_proof_materializes_complete_round(
    root: Path,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": _optional_project_prefix_v1(),
            },
                "constraints": {"op": "all", "children": [_leaf(
                    "PRED-OPTIONAL-PROJECT-POLICY-ACTIONS",
                    "auction.contract",
                    {"contract": "sun"},
                ), _leaf("PRED-OPTIONAL-PROJECT-POLICY-ROOT-COUNT",
                         "proof.winning_root_action_count", {"minimum": 2, "maximum": 2})]},
            "decisionPoint": {
                "kind": "project",
                "selector": {"owner": "east", "projectIndex": 0},
            },
            "outcome": _leaf(
                "PRED-OPTIONAL-PROJECT-POLICY-OUTCOME",
                "outcome.round_margin",
                {
                    "team": "west_east",
                    "minimum": -100000,
                    "maximum": 100000,
                },
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": -1,
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V4-OPTIONAL-PROJECT",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    ).execute(
        scenario,
        proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=8,
        maximum_seconds=90,
    )

    assert result.status == "qualified", result.execution_receipt
    assert result.proof["completionStatus"] == "policy_complete"
    assert {row["action"].split(":", 1)[0] for row in result.proof["rootActions"]} == {
        "pass",
        "declare",
    }
    assert len(result.candidate.trace["completeExactTracePrefix"]["playActions"]) == 32
    assert result.candidate.trace["completeExactNormalization"]["phase"] == "scored"


@pytest.mark.parametrize("mode", ("exact", "bounded"))
def test_optional_project_actor_information_respects_world_scope(
    root: Path,
    mode: str,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": _optional_project_prefix_v1(),
            },
                "constraints": _leaf(
                    "PRED-OPTIONAL-PROJECT-INFO-ACTIONS",
                    "auction.contract",
                    {"contract": "sun"},
            ),
            "decisionPoint": {
                "kind": "project",
                "selector": {"owner": "east", "projectIndex": 0},
            },
                "outcome": _leaf(
                    "PRED-OPTIONAL-PROJECT-INFO-REPORT",
                    "auction.contract",
                    {"contract": "sun"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": mode,
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count",
            "targetTeam": "west_east",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "budgetProfile": "actor_information_set_exact_v1",
            **({"horizon": {"unit": "plies", "value": 2},
                "boundaryEvaluator": "current_raw_point_margin_v1"}
               if mode == "bounded" else {}),
        }
    )

    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V4-OPTIONAL-PROJECT",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    ).execute(
        scenario,
        proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=8,
        maximum_hidden_worlds=1 if mode == "bounded" else 64,
        maximum_seconds=30,
    )

    if mode == "bounded":
        assert result.proof is not None, result.execution_receipt
        assert result.status == "qualified"
        assert result.proof["completionStatus"] == "bounded_complete"
        assert result.proof["hiddenWorlds"]["complete"] is False
        assert result.proof["hiddenWorlds"]["count"] == 1
        return
    assert result.status == "incomplete"
    incomplete = result.execution_receipt["incompleteCompilation"]
    assert incomplete["stage"] == "hidden_world_nonplay_preflight"
    assert incomplete["exhaustionReason"] == "maximum_hidden_worlds"
    assert incomplete["evidence"]["hiddenWorldUpperBound"] > 64


def test_optional_project_continuation_count_reports_depth_exhaustion(
    root: Path,
) -> None:
    rules = RuleBook.load(
        root / "configs/rules/saudi_project_optional_unratified_v1.yaml"
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": _optional_project_prefix_v1(),
            },
                "constraints": _leaf(
                    "PRED-OPTIONAL-PROJECT-LINES-ACTIONS",
                    "auction.contract",
                    {"contract": "sun"},
            ),
            "decisionPoint": {
                "kind": "project",
                "selector": {"owner": "east", "projectIndex": 0},
            },
                "outcome": _leaf(
                    "PRED-OPTIONAL-PROJECT-LINES-REPORT",
                    "auction.contract",
                    {"contract": "sun"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "winning_continuation_line_count",
            "targetTeam": "west_east",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "lineUniverse": "all_legal",
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V4-OPTIONAL-PROJECT",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    ).execute(
        scenario,
        proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=8,
        maximum_seconds=20,
    )

    assert result.status == "incomplete"
    incomplete = result.execution_receipt["incompleteCompilation"]
    assert incomplete["stage"] == "graph"
    assert incomplete["exhaustionReason"] == "maximum_depth"
    cause = incomplete["evidence"]["causeReceipt"]
    assert cause["exhaustionReason"] == "maximum_depth"
    assert cause["sha256"] == canonical_sha256({key: value for key, value in cause.items() if key != "sha256"})


def _scenario(
    *,
    constraints: dict,
    outcome: dict,
    origin: dict | None = None,
    decision_point: dict | None = None,
) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": origin or {"kind": "search"},
            "constraints": constraints,
            "decisionPoint": decision_point
            or {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": outcome,
        }
    )


@pytest.mark.parametrize("tactic", ["large_card_sacrifice", "small_card_big_gain"])
@pytest.mark.parametrize("origin", ["search", "exact_trace_prefix"])
def test_native_causal_admission_precedes_construction_and_preserves_legacy(root, tactic, origin):
    scenario = _scenario(
        origin={"kind": origin, **({"exactTracePrefix": json.loads((
            root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8"))["exactTracePrefix"]} if origin == "exact_trace_prefix" else {})},
        constraints={"op": "not", "children": [_leaf(
            "PRED-CAUSAL", "tactic.verified", {
                "kind": tactic, "causalRequired": True, "minimumScore": 0, "minimumEvidence": 0,
            },
        )]},
        outcome=_leaf("PRED-OUTCOME", "outcome.team_round",
                      {"team": "north_south", "status": "won"}, "report"),
    )
    code = "tactic_causal_verification_unavailable"
    native = OrToolsResearchCompilerV1(native_cp_sat_authority=True).preflight(scenario)
    assert native.unsupported == ({"predicateId": "PRED-CAUSAL", "code": code},)
    assert not OrToolsResearchCompilerV1().preflight(scenario).unsupported
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 1,
        "budgetProfile": "late_game_exact_v1",
    })
    result = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-CAUSAL-PREFLIGHT", bundle_hash="a" * 64,
    ).execute(scenario, request, maximum_seconds=1)
    assert result.status == "incomplete" and result.candidate is None
    assert {item["code"] for item in result.preflight.unsupported} == {code}


def test_native_preflight_admits_bid_state_from_auction_model_only():
    scenario = _scenario(
        constraints=_leaf(
            "PRED-BID-LEGAL-COUNT", "state.legal_action_count",
            {"minimum": 1, "maximum": 8},
        ),
        outcome=_leaf(
            "PRED-BID-REPORT", "state.phase", {"state": "bidding"}, "report",
        ),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    preflight = OrToolsResearchCompilerV1(
        native_cp_sat_authority=True,
    ).preflight(scenario)
    assert preflight.supported

    match_score = _scenario(
        constraints=_leaf(
            "PRED-BID-MATCH", "state.match_score",
            {"team": "north_south", "minimum": 0, "maximum": 152},
        ),
        outcome=_leaf(
            "PRED-BID-REPORT", "state.phase", {"state": "bidding"}, "report",
        ),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    preflight = OrToolsResearchCompilerV1(
        native_cp_sat_authority=True,
    ).preflight(match_score)
    assert preflight.supported

    unsupported = _scenario(
        constraints=_leaf(
            "PRED-BID-RAW", "state.current_raw_points",
            {"team": "north_south", "minimum": 0, "maximum": 152},
        ),
        outcome=_leaf(
            "PRED-BID-REPORT", "state.phase", {"state": "bidding"}, "report",
        ),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    preflight = OrToolsResearchCompilerV1(
        native_cp_sat_authority=True,
    ).preflight(unsupported)
    assert preflight.supported

    bid_game_points = _scenario(
        constraints=_leaf(
            "PRED-BID-GAME", "state.current_game_points",
            {"team": "north_south", "minimum": 0, "maximum": 152},
        ),
        outcome=_leaf(
            "PRED-BID-REPORT", "state.phase", {"state": "bidding"}, "report",
        ),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    preflight = OrToolsResearchCompilerV1(
        native_cp_sat_authority=True,
    ).preflight(bid_game_points)
    assert preflight.supported

    project = _scenario(
        constraints=_leaf("PRED-PROJECT-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf(
            "PRED-PROJECT-ACTOR", "state.next_actor", {"seat": "north"}, "report",
        ),
        decision_point={"kind": "project", "selector": {"owner": "north"}},
    )
    assert OrToolsResearchCompilerV1(
        native_cp_sat_authority=True,
    ).preflight(project).supported
    count = _scenario(
        constraints=_leaf(
            "PRED-PROJECT-COUNT", "state.legal_action_count",
            {"minimum": 1, "maximum": 64},
        ),
        outcome=_leaf(
            "PRED-PROJECT-ACTOR", "state.next_actor", {"seat": "north"}, "report",
        ),
        decision_point={"kind": "project", "selector": {"owner": "north"}},
    )
    assert OrToolsResearchCompilerV1(
        native_cp_sat_authority=True,
    ).preflight(count).supported


@pytest.mark.parametrize("kind", (
    "state.current_raw_points",
    "state.current_project_points",
    "state.current_game_points",
))
def test_native_bid_nonzero_score_is_rejected_during_construction(rules, kind):
    scenario = _scenario(
        constraints=_leaf(
            "PRED-BID-NONZERO-SCORE", kind,
            {"team": "north_south", "minimum": 1, "maximum": 152},
        ),
        outcome=_leaf(
            "PRED-BID-PHASE", "state.phase", {"state": "bidding"}, "report",
        ),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, native_cp_sat_scoring=True
    )(scenario, (), seed=0, maximum_seconds=10)
    assert attempt.status == "infeasible", attempt.reason
    assert attempt.candidate is None


def test_native_bid_zero_score_cut_preserves_other_any_branch(rules):
    scenario = _scenario(
        constraints={"op": "any", "children": [
            _leaf("PRED-BID-NONZERO", "state.current_game_points",
                  {"team": "north_south", "minimum": 1, "maximum": 152}),
            _leaf("PRED-BID-SUN", "auction.contract", {"contract": "sun"}),
        ]},
        outcome=_leaf(
            "PRED-BID-PHASE", "state.phase", {"state": "bidding"}, "report",
        ),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, native_cp_sat_scoring=True
    )(scenario, (), seed=0, maximum_seconds=15)
    assert attempt.status == "feasible", attempt.reason
    assert attempt.candidate is not None
    assert attempt.candidate.trace["contract"] == "sun"
    auction_receipt = attempt.candidate.stage_receipts["auction"]
    assert any(
        row["predicateId"] == "PRED-BID-NONZERO"
        for row in auction_receipt.evidence_bindings
    )
    assert any(
        row["literalName"].startswith("assume_bid_zero_score_")
        for row in auction_receipt.named_assumptions
    )


def test_unfixed_play_position_selects_actor_anywhere_in_trick():
    scenario = _scenario(
        constraints=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        outcome=_leaf("PRED-OUTCOME", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "remainingCards": 2, "actingSeat": "south",
        }},
    )
    plays = [SimpleNamespace(actor=seat) for _ in range(8)
             for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH, Seat.EAST)]
    assert executor_module._static_decision_prefix_length(scenario) is None
    assert executor_module._decision_prefix_length(scenario, plays) == 26

    fixed = _scenario(
        constraints=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        outcome=_leaf("PRED-OUTCOME", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
            "actingSeat": "north",
        }},
    )
    assert executor_module._static_decision_prefix_length(fixed) == 24
    assert executor_module._decision_prefix_length(fixed, plays) == 24

    leader = _scenario(
        constraints=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        outcome=_leaf("PRED-OUTCOME", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {"trickIndex": 6}},
    )
    assert executor_module._static_decision_prefix_length(leader) == 24
    assert executor_module._decision_prefix_length(leader, plays) == 24


def test_play_decision_derives_actor_for_remaining_card_count():
    scenario = _scenario(
        constraints=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        outcome=_leaf("PRED-OUTCOME", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "remainingCards": 2,
        }},
    )
    plays = [SimpleNamespace(actor=seat) for _ in range(8)
             for seat in (Seat.NORTH, Seat.WEST, Seat.SOUTH, Seat.EAST)]
    assert executor_module._decision_prefix_length(scenario, plays) == 24


def test_native_same_solve_never_qualifies_a_divergent_proof_continuation():
    scenario = _scenario(
        constraints=_leaf(
            "PRED-PLAY", "play.action_matches",
            {"trickIndex": 5, "position": 0, "selector": {"ranks": ["ace"]}},
        ),
        outcome=_leaf("PRED-OUTCOME", "outcome.kabout", {"team": "north_south", "achieved": False}),
    )
    source_hash = "a" * 64
    model_hash = "b" * 64
    evidence = {}
    bindings = []
    for predicate in scenario.predicates():
        core = {
            "schemaVersion": 1,
            "predicateId": predicate.predicate_id,
            "kind": predicate.kind,
            "satisfied": True,
            "evidenceExtractor": "cp_sat.test_same_solve",
            "loweringHash": "c" * 64,
            "observed": {},
        }
        row = {**core, "sha256": canonical_sha256(core)}
        evidence[predicate.predicate_id] = row
        bindings.append({
            "predicateId": predicate.predicate_id,
            "evidenceHash": row["sha256"],
            "sourceEvidenceHash": source_hash,
            "modelHash": model_hash,
        })
    receipt = SimpleNamespace(
        evidence_bindings=tuple(bindings),
        evidence_hash=source_hash,
        model_hash=model_hash,
        to_dict=lambda: {"sha256": "d" * 64},
    )
    trace = {"play": {"plays": [{"card": "SA"}]}}
    candidate = SimpleNamespace(trace=trace, stage_receipts={"tailConstruction": receipt})
    assert executor_module._native_same_solve_evaluation_v3(
        scenario, candidate, None, evidence
    ).satisfied
    candidate.trace = {
        **trace,
        "completePlay": {"plays": [{"card": "HK"}]},
        "proofContinuation": {"kind": "play_exact_principal_variation_continuation_v1"},
    }
    assert executor_module._native_same_solve_evaluation_v3(
        scenario, candidate, None, evidence
    ) is None


def _proof(
    *,
    claim: str = "outcome_reachable",
    selected_action: str | None = None,
    target_team: str = "north_south",
) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "complete_state",
        "claim": claim,
        "targetTeam": target_team,
        "utility": "round_win",
        "successThreshold": 1,
        "budgetProfile": "test_exact",
    }
    if selected_action is not None:
        value["selectedAction"] = selected_action
    return ScenarioProofRequestV1.build(value)


def _bounded_proof(
    *,
    claim: str = "outcome_reachable",
    selected_action: str | None = None,
    target_team: str = "north_south",
) -> ScenarioProofRequestV1:
    value = {
        "schemaVersion": 1,
        "mode": "bounded",
        "visibility": "complete_state",
        "claim": claim,
        "targetTeam": target_team,
        "utility": "round_win",
        "successThreshold": 1,
        "horizon": {"unit": "plies", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
        "budgetProfile": "test_bounded",
    }
    if selected_action is not None:
        value["selectedAction"] = selected_action
    return ScenarioProofRequestV1.build(value)


def _policy_templates() -> dict:
    result = {}
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
        result[seat] = {key: value for key, value in core.items() if key != "seat"}
        result[seat]["profileHash"] = canonical_sha256(core)
    return result


@pytest.mark.real_clock
@pytest.mark.parametrize("origin_kind", ["search", "exact_trace_prefix"])
@pytest.mark.parametrize("backend_id", ["ortools_cp_sat_v2", "ortools_cp_sat_v3"])
def test_shard_campaign_deadline_reaches_both_origins(root, monkeypatch, origin_kind, backend_id):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    scenario = _scenario(
        constraints=_leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-OUTCOME", "outcome.team_round",
            {"team": "north_south", "status": "won"}, "report"),
        origin={"kind": origin_kind, **({"exactTracePrefix": _optional_project_prefix_v1()}
            if origin_kind == "exact_trace_prefix" else {})},
    )
    seen = []
    campaign = executor_module.OrToolsResearchCampaignResultV1(
        "incomplete", (), (), (), (), 1, 1,
        "rules_preserving_v1", "decision_state_card_hamming_v1", 4, 0.0, 0.0)
    def seed_campaign(self, _scenario, _proof_request, **kwargs):
        seen.append(kwargs)
        return campaign
    def process_campaign(task, _request, **_kwargs):
        seen.append(task["options"])
        return campaign
    monkeypatch.setattr(
        executor_module,
        "_run_seed_campaign_process_v1",
        process_campaign,
    ) if backend_id == "ortools_cp_sat_v3" else monkeypatch.setattr(
        OrToolsUnifiedResearchExecutorV1,
        "_execute_seed_campaign",
        seed_campaign,
    )
    shards = partition_research_shards(
        run_code="RUN-DEADLINE001", run_hash=canonical_sha256("deadline-run"),
        recipe_hash=scenario.sha256, seed_start=0, seed_count=1,
        seeds_per_shard=1, maximum_attempts=1,
    )
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id=backend_id,
        bundle_revision_id="CBR-TEST-DEADLINE", bundle_hash="a" * 64)
    executor.execute_campaign(scenario, _bounded_proof(), result_count=1,
        symmetry_policy_id="rules_preserving_v1", allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4, maximum_candidates=1,
        maximum_seconds=180, maximum_campaign_seconds=60, shards=shards,
        shard_worker_count=1)
    assert len(seen) == 1
    assert seen[0]["maximum_seconds"] == 180
    if backend_id == "ortools_cp_sat_v3":
        assert 0 < seen[0]["maximum_campaign_seconds"] <= 60
    else:
        assert "maximum_campaign_seconds" not in seen[0]


@pytest.mark.real_clock
@pytest.mark.parametrize(
    "seed_count,worker_count,campaign_seconds",
    [(2, 1, 60), (8, 2, 300)],
)
def test_shards_use_remaining_campaign_time_without_reducing_per_solve_allowance(
    root, monkeypatch, seed_count, worker_count, campaign_seconds,
):
    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    scenario = _scenario(
        constraints=_leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-OUTCOME", "outcome.team_round",
            {"team": "north_south", "status": "won"}, "report"),
    )
    seen = []
    clock = [0.0]
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(**{
        **vars(executor_module.time), "perf_counter": lambda: clock[0],
    }))

    def process_campaign(task, _request, **_kwargs):
        options = task["options"]
        seen.append((options["maximum_seconds"], options["maximum_campaign_seconds"]))
        clock[0] += 5
        return executor_module.OrToolsResearchCampaignResultV1(
            "incomplete", (), (), (), (), 1, 1,
            "rules_preserving_v1", "decision_state_card_hamming_v1", 4, 0.0, 0.0)

    monkeypatch.setattr(
        executor_module,
        "_run_seed_campaign_process_v1",
        process_campaign,
    )
    shards = partition_research_shards(
        run_code="RUN-DEADLINE002", run_hash=canonical_sha256("deadline-run-2"),
        recipe_hash=scenario.sha256, seed_start=0, seed_count=seed_count,
        seeds_per_shard=1, maximum_attempts=1,
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-TEST-DEADLINE", bundle_hash="a" * 64)

    executor.execute_campaign(
        scenario, _bounded_proof(), result_count=1,
        symmetry_policy_id="rules_preserving_v1", allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4, maximum_candidates=seed_count,
        maximum_seconds=30, maximum_campaign_seconds=campaign_seconds, shards=shards,
        shard_worker_count=worker_count)

    assert len(seen) == seed_count
    assert all(solve_seconds == 30 for solve_seconds, _campaign_seconds in seen)
    assert all(0 < seconds <= campaign_seconds for _solve_seconds, seconds in seen)
    assert seen[0][1] == campaign_seconds
    if worker_count == 1:
        assert seen[1][1] == campaign_seconds - 5


def _executor(rules) -> OrToolsUnifiedResearchExecutorV1:
    return OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V4",
        bundle_hash="a" * 64,
    )


@pytest.fixture(scope="module")
def complete_candidate(rules) -> OrToolsNeutralCandidateV1:
    scenario = _scenario(
        constraints=_leaf(
            "PRED-SEARCH-SEED",
            "deal.purchase_suit",
            {"suit": "diamonds"},
            "prefer",
        ),
        outcome=_leaf(
            "PRED-SEARCH-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )
    attempt = GenericCompleteTraceCompilerV1(rules=rules)(
        scenario,
        (),
        seed=0,
        maximum_seconds=30,
    )
    assert attempt.status == "feasible", (attempt.reason, attempt.partial_receipt)
    assert attempt.candidate is not None
    return attempt.candidate


@pytest.mark.parametrize("source", ["ordinary", "construction", "partial_root", "continuation", "missing_escalation"])
def test_native_outcome_keeps_terminal_sources_together(rules, complete_candidate, monkeypatch, source):
    trace = copy.deepcopy(dict(complete_candidate.trace))
    terminal_play, terminal_score = trace["play"], trace["score"]
    terminal_buyer = Seat.parse(trace["buyer"])
    trace["matchScore"] = [101, 30]
    if source != "ordinary":
        trace.update(constructionPlay=terminal_play, constructionScore=terminal_score,
            constructionEscalation={"observation": {"level": "double", "actions": []}},
            play={"trickWinners": ["west"]} if source == "partial_root" else None,
            escalation={"observation": {"level": "normal", "actions": []}},
            buyer=None, contract=None, score={"rawPoints": [-1, -1]})
    if source == "continuation":
        trace.update(completePlay=terminal_play, score=terminal_score,
            completeExactNormalization={"state": {"buyer": terminal_buyer.name.lower(),
                "contract": terminal_score["contract"], "escalation": "double",
                "escalationHistory": [], "matchScore": [999, 999]}},
            constructionPlay={"trickWinners": []}, constructionScore={"rawPoints": [-1, -1]})
    if source == "missing_escalation":
        trace.pop("constructionEscalation")
        with pytest.raises(ValueError, match="terminal escalation"):
            executor_module._native_outcome_context_v3(SimpleNamespace(trace=trace), rules, terminal_buyer)
        return
    observed = []
    original = executor_module.OrToolsScoringStageV1.solve

    def solve(stage, **kwargs):
        observed.append(kwargs)
        return original(stage, **kwargs)

    monkeypatch.setattr(executor_module.OrToolsScoringStageV1, "solve", solve)
    context = executor_module._native_outcome_context_v3(SimpleNamespace(trace=trace), rules,
        terminal_buyer if source == "ordinary" else terminal_buyer.next())
    assert observed[0]["buyer"] == terminal_buyer
    assert observed[0]["contract"].value == terminal_score["contract"]
    assert observed[0]["escalation"] == (
        trace["escalation"]["observation"]["level"] if source == "ordinary" else "double")
    assert observed[0]["raw_points"] == tuple(terminal_score["rawPoints"])
    assert observed[0]["starting_match_score"] == (101, 30)
    assert len(observed[0]["trick_winner_teams"]) == 8
    assert context.source_stage == "scoring"


@pytest.mark.parametrize("native", [False, True])
def test_native_bid_root_keeps_construction_outcome_without_root_buyer(rules, complete_candidate, native):
    from baloot_platform.research.construction_trace import construction_trace_prefix_v1

    raw = construction_trace_prefix_v1(complete_candidate.trace).to_dict()
    raw.pop("sha256")
    raw.update(auctionActions=[], escalationActions=[], projectActions=[],
        projectDeclarations=[], playActions=[])
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid", "selector": {"actionIndex": 0}})
    candidate = executor_module._rebind_search_candidate_to_decision(
        complete_candidate, prefix, normalized, preserve_construction_outcome=True, native_cp_sat=native)
    assert candidate.trace["buyer"] is candidate.trace["contract"] is candidate.trace["play"] is None
    if not native:
        assert "constructionEscalation" not in candidate.trace
        return
    bindings = DefaultStageContextCompilerV1(rules, native_cp_sat_authority=True,
        required_stages=("state", "outcome"))(candidate, None)
    assert "outcome" in bindings
    assert candidate.trace["constructionEscalation"] == complete_candidate.trace["escalation"]
    solver = cp_model.CpSolver()
    outcome = bindings["outcome"].context
    assert solver.solve(outcome.model) == cp_model.OPTIMAL
    assert solver.value(outcome.buyer_seat) == int(Seat.parse(complete_candidate.trace["buyer"]))
    assert [solver.value(value) for value in outcome.game_points] == complete_candidate.trace["score"]["gamePoints"]


@pytest.mark.parametrize("stages", [("deal",), ("deal", "auction")])
@pytest.mark.parametrize("contract", ["sun", "ashkal"])
@pytest.mark.parametrize("origin", ["search", "exact_trace_prefix"])
@pytest.mark.parametrize("recipient", [Seat.NORTH, Seat.WEST])
def test_v3_remaining_deal_ownership_uses_native_purchase_and_prefix(root, stages, contract, origin, recipient):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    deck = list(range(32))
    purchase = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST).purchase_position
    deck[purchase], deck[0] = deck[0], deck[purchase]
    caller = recipient if contract == "sun" else recipient.partner
    actions = ["pass"] * int(caller) + [contract] + ["pass"] * int(caller)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=recipient)
    owners = {deck[position]: seat for seat in Seat
              for position in layout.positions_for(seat, final=True)}
    played = deck[layout.positions_for(Seat.WEST, final=True)[0]]
    future = deck[layout.positions_for(Seat.SOUTH, final=True)[0]]
    candidate = SimpleNamespace(origin=origin, state={"contract": "sun"}, stage_receipts={
        stage: SimpleNamespace(evidence_hash="a" * 64) for stage in stages}, trace={
        "deckOrder": deck,
        "auction": {"completion": {"actions": [{"action": action} for action in actions]}},
        "play": {"plays": [{"actor": "west", "card": played},
                            {"actor": "south", "card": future}],
                 "decisionState": {"prefixLength": 1,
                                   "remainingHands": {"east": list(range(32))}}},
    })
    if origin == "exact_trace_prefix":
        candidate.trace["exactTracePrefix"] = {
            "deckOrder": deck, "dealer": "east",
            "auctionActions": [{"action": action} for action in actions],
            "playActions": [{"actor": "west", "card": played}],
        }
    compiler = DefaultStageContextCompilerV1(rules, required_stages=stages,
                                           native_cp_sat_authority=True)
    context = compiler(candidate, None)["deal"].context
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert solver.value(context.scenario.purchase_recipient) == int(recipient)
    assert solver.value(context.scenario.buyer) == int(
        recipient.partner if contract == "ashkal" else recipient)
    assert [solver.value(owner) for owner in context.owners("remaining")] == [
        4 if card == played else int(owners[card]) for card in range(32)]
    assert solver.value(context.owners("remaining")[future]) == int(Seat.SOUTH)

    candidate.trace["play"]["plays"][0]["actor"] = "east"
    malformed = compiler(candidate, None)["deal"].context
    assert solver.solve(malformed.model) == cp_model.INFEASIBLE


def test_v3_remaining_deal_ownership_ignores_generated_candidate_header(rules, complete_candidate):
    candidate = replace(complete_candidate, trace=copy.deepcopy(complete_candidate.trace))
    expected = {Card.parse(card).id: int(Seat.parse(seat))
                for seat, cards in candidate.trace["play"]["decisionState"]["remainingHands"].items()
                for card in cards}
    candidate.trace["play"]["decisionState"]["remainingHands"] = {"east": list(range(32))}
    context = DefaultStageContextCompilerV1(rules, required_stages=("deal",),
        native_cp_sat_authority=True)(candidate, None)["deal"].context
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert [solver.value(owner) for owner in context.owners("remaining")] == [
        expected.get(card, 4) for card in range(32)]


@pytest.mark.parametrize("structural,projects,expected", [
    pytest.param(True, (("project.exists", {"team": "north_south", "kind": "siri", "value": 4, "state": "eligible"}),
                       ("project.exists", {"team": "west_east", "kind": "siri", "value": 4, "state": "eligible"})), True, id="eligible-exists"),
    pytest.param(True, (("project.count", {"team": "north_south", "minimum": 0, "maximum": 4}),), True, id="eligible-count-default"),
    pytest.param(True, (("project.combined_value", {"team": "north_south", "minimum": 0, "maximum": 40, "state": "eligible"}),), True, id="eligible-combined-value"),
    pytest.param(True, (("project.overlap_count", {"owner": "north", "minimum": 0, "maximum": 8}),), True, id="eligible-overlap"),
    pytest.param(True, (("project.count", {"team": "north_south", "minimum": 0, "maximum": 4, "state": "awarded"}),), False, id="awarded-count"),
    pytest.param(True, (("project.combined_value", {"team": "north_south", "minimum": 0, "maximum": 40, "state": "declared"}),), False, id="declared-value"),
    pytest.param(True, (("project.baloot", {"team": "north_south", "achieved": False}),), False, id="baloot"),
    pytest.param(True, (("project.overlap_count", {"owner": "north", "minimum": 0, "maximum": 8}),
                       ("project.exists", {"team": "west_east", "kind": "siri", "value": 4, "state": "awarded"})), False, id="mixed-eligible-awarded"),
    pytest.param(False, (("project.exists", {"team": "north_south", "kind": "siri", "value": 4, "state": "eligible"}),), False, id="nonstructural"),
])
def test_native_eligibility_projection_dispatch_requires_every_project_leaf(root, monkeypatch, structural, projects, expected):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")

    def observe_fragment(*args, **kwargs):
        assert kwargs.get("eligibility_only", False) is expected
        raise RuntimeError("projection-dispatch-observed")

    monkeypatch.setattr(executor_module, "attach_project_lifecycle_fragment_v1", observe_fragment)
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        {"op": "any", "children": [_leaf(f"PRED-PROJECT-{i}", kind, arguments)
            for i, (kind, arguments) in enumerate(projects)]},
    ]}, outcome=_leaf("PRED-REPORT", "state.legal_action_count", {"minimum": 0, "maximum": 32}, "report"))
    with pytest.raises(RuntimeError, match="projection-dispatch-observed"):
        GenericCompleteTraceCompilerV1(rules=rules, structural_only=structural,
            native_cp_sat_scoring=True,
            manifest=executor_module.shared_scenario_predicate_catalog_lowering_manifest_v3())(
                scenario, (), seed=0, maximum_seconds=10)


@pytest.mark.parametrize("contract", ["sun", "hokum"])
@pytest.mark.parametrize("has_projects", [True, False])
def test_v8_structural_eligible_count_value_overlap_use_known_ownership(root, contract, has_projects):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    hands = {
        "north": "S7 S8 S9 S10 SJ H7 H9 HJ",
        "west": "SQ SK SA H8 H10 HQ HK HA",
        "south": "D7 D9 DJ DK C7 C9 CJ CK",
        "east": "D8 D10 DQ DA C8 C10 CQ CA",
    }
    ownership = {Card.parse(card).id: seat for seat, cards in hands.items() for card in cards.split()}
    assert len(ownership) == 32
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-CONTRACT", "auction.contract", {"contract": contract}),
        _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
        *([_leaf("PRED-TRUMP", "auction.trump", {"suit": "spades"})] if contract == "hokum" else []),
        *[_leaf(f"PRED-OWNER-{card}", "deal.card_owner", {"card": card,
            "seat": seat, "handStage": "post_purchase"}) for card, seat in ownership.items()],
        _leaf("PRED-COUNT", "project.count", {"team": "north_south",
            "minimum": 1 if has_projects else 0, "maximum": 100 if has_projects else 0,
            "state": "eligible"}),
        _leaf("PRED-VALUE", "project.combined_value", {"team": "north_south",
            "minimum": 1 if has_projects else 0, "maximum": 1000, "state": "eligible"}),
        _leaf("PRED-OVERLAP", "project.overlap_count", {"owner": "north",
            "minimum": 1 if has_projects else 0, "maximum": 8}),
    ]}, outcome=_leaf("PRED-REPORT", "state.legal_action_count",
        {"minimum": 0, "maximum": 32}, "report"))
    attempt = GenericCompleteTraceCompilerV1(rules=rules, structural_only=True,
        native_cp_sat_scoring=True,
        manifest=executor_module.shared_scenario_predicate_catalog_lowering_manifest_v3())(
            scenario, (), seed=0, maximum_seconds=15)
    assert attempt.status == ("feasible" if has_projects else "infeasible"), (attempt.reason, attempt.partial_receipt)
    if has_projects:
        trace = attempt.candidate.trace
        layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.parse(trace["dealer"]), buyer=Seat.NORTH)
        solved_owners = {trace["deckOrder"][position]: seat.name.lower()
            for seat in Seat for position in layout.positions_for(seat, final=True)}
        assert solved_owners == ownership
        assert trace["constructionScope"] == "deal_auction_project_ownership"
        assert "play" not in trace
    else:
        assert attempt.candidate is None
        assert attempt.reason == "deal_auction_project_infeasible"


@pytest.mark.parametrize("contract,value", [("sun", 4), ("hokum", 2)])
def test_native_structural_eligible_projects_reach_real_solver(rules, monkeypatch, contract, value):
    def forbid_witness_enumeration(*args, **kwargs):
        raise AssertionError("eligibility-only construction must solve all patterns jointly")

    monkeypatch.setattr(executor_module, "_project_exists_witness_keys", forbid_witness_enumeration)
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-CONTRACT", "auction.contract", {"contract": contract}),
        *([_leaf("PRED-TRUMP", "auction.trump", {"suit": "diamonds"}),
            *[_leaf(f"PRED-TRUMP-COUNT-{seat.upper()}", "deal.suit_count", {"seat": seat,
                "suit": "diamonds", "handStage": "post_purchase", "minimum": 2, "maximum": 2})
                for seat in ("north", "west", "south", "east")]] if contract == "hokum" else []),
        *[_leaf(f"PRED-PROJECT-{index}", "project.exists", {"team": team,
            "kind": "siri", "value": value, "state": "eligible"})
            for index, team in enumerate(("north_south", "west_east"))],
    ]}, outcome=_leaf("PRED-REPORT", "state.legal_action_count", {"minimum": 0, "maximum": 32}, "report"))
    attempt = GenericCompleteTraceCompilerV1(rules=rules, structural_only=True,
        native_cp_sat_scoring=True,
        manifest=executor_module.shared_scenario_predicate_catalog_lowering_manifest_v3())(
            scenario, (), seed=0, maximum_seconds=10)
    assert attempt.status == "feasible", (attempt.reason, attempt.partial_receipt)
    assert attempt.candidate.trace["constructionScope"] == "deal_auction_project_ownership"
    assert "play" not in attempt.candidate.trace


def test_quality_structural_construction_does_not_build_a_competing_play(
    rules, monkeypatch
):
    monkeypatch.setattr(
        executor_module.OrToolsEscalationStageV1,
        "solve",
        lambda *args, **kwargs: pytest.fail("structural proposal solved escalation"),
    )
    monkeypatch.setattr(
        executor_module.OrToolsProjectLifecycleModelV1,
        "solve",
        lambda *args, **kwargs: pytest.fail("structural proposal re-solved projects"),
    )
    monkeypatch.setattr(
        OrToolsPlayTailStageV1,
        "solve",
        lambda *args, **kwargs: pytest.fail("structural proposal built a play line"),
    )
    scenario = _scenario(
        constraints=_leaf(
            "PRED-STRUCTURAL-SEED",
            "deal.purchase_suit",
            {"suit": "diamonds"},
            "prefer",
        ),
        outcome=_leaf(
            "PRED-STRUCTURAL-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )

    attempt = GenericCompleteTraceCompilerV1(
        rules=rules,
        structural_only=True,
    )(scenario, (), seed=0, maximum_seconds=15)

    assert attempt.status == "feasible"
    assert attempt.candidate is not None
    assert attempt.candidate.trace["constructionScope"] == "deal_auction_project_ownership"
    assert "play" not in attempt.candidate.trace
    assert set(attempt.candidate.stage_receipts) == {"structural"}
    assert len(attempt.candidate.no_good_vector) == 32


@pytest.mark.parametrize("kind,arguments,hardness,backend", [
    ("auction.contract", {"contract": "sun"}, "hard", "ortools_cp_sat_v3"),
    ("auction.buyer", {"seat": "north"}, "hard", "ortools_cp_sat_v3"),
    ("auction.action_sequence", {"actions": [{"action": "sun"}], "complete": True}, "hard", "ortools_cp_sat_v3"),
    ("auction.action_sequence", {"actions": [{"action": "pass"}], "complete": False}, "hard", "ortools_cp_sat_v3"),
    ("auction.action_sequence", {"actions": [{"action": "sun"}], "complete": True}, "prefer", "ortools_cp_sat_v3"),
    ("auction.action_sequence", {"actions": [{"action": "sun"}], "complete": True}, "report", "ortools_cp_sat_v3"),
    ("auction.contract", {"contract": "sun"}, "hard", "ortools_cp_sat_v2"),
    (None, None, None, "ortools_cp_sat_v3"),
])
def test_quality_generation_preserves_the_construction_auction(complete_candidate, kind, arguments, backend, hardness):
    prefix = executor_module._construction_auction_prefix_v1(complete_candidate)
    expected = complete_candidate.trace["auction"]["completion"]["actions"]

    assert [row["action"] for row in prefix.auction_actions] == [
        row["action"] for row in expected
    ]
    assert prefix.auction_actions
    assert prefix.project_actions_present and not prefix.project_actions
    assert not prefix.play_actions

    scenario = _scenario(
        constraints=_leaf("PRED-AUCTION", kind, arguments, hardness),
        outcome=_leaf("PRED-OUTCOME", "outcome.team_round",
                      {"team": "north_south", "status": "won"}, "report"),
    ) if hardness is not None else None
    owner = SimpleNamespace(backend_id=backend, _resolve_quality_candidate=lambda *args, **kwargs: kwargs)
    call = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
        owner, complete_candidate, scenario=scenario, maximum_seconds=30)
    selecting = backend == "ortools_cp_sat_v3" and hardness is not None and not (
        kind == "auction.action_sequence" and hardness == "hard")
    assert call["recorded_trace"] == (None if selecting else prefix)
    assert call["recorded_events"] == (None if selecting else ())
    assert call["continue_recorded_prefix"] is not selecting
    assert call["scenario"] is scenario and call["maximum_seconds"] == 30
    exact_call = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
        owner, replace(complete_candidate, origin="exact_trace_prefix"),
        scenario=scenario, maximum_seconds=30)
    assert exact_call["recorded_trace"] == prefix
    assert exact_call["continue_recorded_prefix"] is True
    nested = _scenario(
        constraints={"op": "any", "children": [
            _leaf("PRED-SEQUENCE", "auction.action_sequence",
                  {"actions": [{"action": "pass"}], "complete": False}),
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"})]},
        outcome=_leaf("PRED-REPORT", "outcome.team_round",
                      {"team": "north_south", "status": "won"}, "report"))
    nested_call = OrToolsUnifiedResearchExecutorV1.generate_quality_candidate(
        owner, complete_candidate, scenario=nested, maximum_seconds=30)
    assert nested_call["recorded_trace"] == prefix


@pytest.mark.parametrize("native_cp_sat", [False, True])
@pytest.mark.parametrize("contradiction", ["contract", "card_owner"])
def test_search_construction_reports_contradictory_constraints_infeasible(
    root, rules, monkeypatch, native_cp_sat, contradiction
):
    if native_cp_sat:
        import baloot_platform.research.ortools_scenario_model as auction_model

        rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")

        def forbidden_legacy_table(*_args, **_kwargs):
            pytest.fail("native contradiction proof must not use legacy auction tables")

        monkeypatch.setattr(auction_model, "_enumerate_auction", forbidden_legacy_table)
    statuses = []
    original_solve = cp_model.CpSolver.solve

    def solve(solver, model, *args, **kwargs):
        status = original_solve(solver, model, *args, **kwargs)
        statuses.append(status)
        return status

    monkeypatch.setattr(cp_model.CpSolver, "solve", solve)
    conflicting = (
        _leaf("PRED-CONTRACT-HOKUM", "auction.contract", {"contract": "hokum"})
        if contradiction == "contract"
        else {"op": "all", "children": [
            _leaf(f"PRED-OWNER-{seat.upper()}", "deal.card_owner",
                  {"card": Card.parse("S7").id, "seat": seat, "handStage": "post_purchase"})
            for seat in ("north", "south")
        ]}
    )
    scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf("PRED-CONTRACT-SUN", "auction.contract", {"contract": "sun"}),
                conflicting,
            ],
        },
        outcome=_leaf(
            "PRED-CONTRACT-REPORT",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )

    compiler = GenericCompleteTraceCompilerV1(
        rules=rules,
        structural_only=True,
        native_cp_sat_scoring=native_cp_sat,
        manifest=(executor_module.shared_scenario_predicate_catalog_lowering_manifest_v3()
                  if native_cp_sat else None),
    )
    attempt = compiler(
        scenario,
        (),
        seed=0,
        maximum_seconds=5,
    )

    assert attempt.status == "infeasible", (attempt.reason, attempt.partial_receipt)
    assert attempt.candidate is None
    assert attempt.reason == "deal_auction_infeasible"
    if contradiction == "card_owner":
        assert cp_model.INFEASIBLE in statuses


@pytest.mark.parametrize("operator,backend", [
    ("all", "ortools_cp_sat_v3"), ("any", "ortools_cp_sat_v3"),
    ("not", "ortools_cp_sat_v3"), ("count", "ortools_cp_sat_v3"),
    ("prefer", "ortools_cp_sat_v3"), ("report", "ortools_cp_sat_v3"),
    ("all", "ortools_cp_sat_v2"),
    ("unknown", "ortools_cp_sat_v3"), ("invalid", "ortools_cp_sat_v3"),
    ("cancelled", "ortools_cp_sat_v3"),
    ("range", "ortools_cp_sat_v3"),
    ("range_overlap", "ortools_cp_sat_v3"), ("range_contradiction", "ortools_cp_sat_v3"),
    ("range_unknown", "ortools_cp_sat_v3"), ("range_local_infeasible", "ortools_cp_sat_v3"),
])
def test_generic_search_receives_only_required_fixed_initial_inputs(root, monkeypatch, operator, backend):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    children = [
        _leaf("PRED-DEALER", "auction.dealer", {"seat": "north"}),
        _leaf("PRED-SCORE-NS", "state.match_score", {"team": "north_south", "minimum": 101, "maximum": 101}),
        _leaf("PRED-SCORE-WE", "state.match_score", {"team": "west_east", "minimum": 30, "maximum": 30}),
    ]
    expression = {"op": "all", "children": children}
    if operator == "any":
        expression = {"op": "any", "children": [expression,
            _leaf("PRED-OTHER", "state.phase", {"state": "bidding"})]}
    elif operator == "not":
        expression = {"op": "not", "children": [children[1]]}
    elif operator == "count":
        expression.update(op="count", minimum=3, maximum=3)
    elif operator.startswith("range"):
        for child in children[1:]:
            child["arguments"]["maximum"] += 2
        if operator in {"range_overlap", "range_contradiction"}:
            children.append(_leaf("PRED-SCORE-NS-SECOND", "state.match_score", {
                "team": "north_south", "minimum": 102 if operator == "range_overlap" else 104,
                "maximum": 105,
            }))
    elif operator in {"prefer", "report"}:
        for child in children:
            child["hardness"] = operator
    scenario = _scenario(constraints=expression,
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "bidding"}, "report"),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}})
    observed = []

    def construct(compiler, *_args, **_kwargs):
        observed.append((compiler.dealer, compiler.match_score))
        return OrToolsCandidateAttemptV1("infeasible" if operator == "range_local_infeasible"
            else "incomplete", None, "dispatch_only")

    monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__", construct)
    cancellation = [False]
    if operator in {"unknown", "invalid"}:
        status = cp_model.UNKNOWN if operator == "unknown" else cp_model.MODEL_INVALID
        monkeypatch.setattr(cp_model.CpSolver, "solve", lambda *_args: status)
    elif operator == "range_unknown":
        original_solve = cp_model.CpSolver.solve
        monkeypatch.setattr(cp_model.CpSolver, "solve", lambda solver, model:
            cp_model.UNKNOWN if any(var.name.startswith("initial_match_score_")
                for var in model.proto.variables) else original_solve(solver, model))
    elif operator == "cancelled":
        def cancel_during_solve(_solver, _model, cancel_check):
            assert cancel_check is not None
            cancellation[0] = True
            return cp_model.INFEASIBLE
        monkeypatch.setattr(executor_module, "_solve_cp_sat_with_cancel_v1", cancel_during_solve)
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id=backend,
        bundle_revision_id="CBR-INITIAL-INPUTS", bundle_hash="a" * 64)
    result = executor.execute(scenario, _proof(), maximum_candidates=1,
        maximum_seconds=5, maximum_proof_nodes=1000, maximum_proof_depth=4,
        cancel_check=(lambda: cancellation[0]) if operator == "cancelled" else None)
    if operator == "range_contradiction":
        assert result.status == "infeasible" and observed == []
        return
    assert result.status == ("not_found" if operator == "range_local_infeasible" else "incomplete")
    if operator in {"unknown", "invalid", "cancelled", "range_unknown"}:
        assert observed == []
        partial = result.execution_receipt["incompleteCompilation"]
        assert partial["stage"] == "initial_state_inputs"
        assert partial["exhaustionReason"] == ("cancelled" if operator == "cancelled" else "expression_projection_incomplete")
        return
    if operator.startswith("range"):
        if operator == "range_local_infeasible":
            assert len(observed) == len(set(observed)) == 9
        else:
            assert len(observed) == 1
        dealer, scores = observed[0]
        assert dealer == Seat.NORTH
        assert (102 if operator == "range_overlap" else 101) <= scores[0] <= 103
        assert 30 <= scores[1] <= 32
        return
    expected = (Seat.NORTH, (101, 30)) if backend == "ortools_cp_sat_v3" and operator in {"all", "count"} else (Seat.EAST, (0, 0))
    assert observed == [expected]


@pytest.mark.parametrize("operator", ["any", "not", "count", "correlated", "contradiction",
    "tied", "dealer_not", "fixed_conflict", "dealer_conflict", "completed"])
def test_generic_search_constructs_composed_initial_inputs(root, monkeypatch, operator):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    north = _leaf("PRED-DEALER-NORTH", "auction.dealer", {"seat": "north"})
    score = _leaf("PRED-SCORE-NS", "state.match_score", {
        "team": "north_south", "minimum": 101, "maximum": 101})
    second = _leaf("PRED-SCORE-NS-SECOND", "state.match_score", {
        "team": "north_south", "minimum": 103, "maximum": 103})
    expression = {"op": "all", "children": [north,
        _leaf("PRED-SCORE-WE", "state.match_score", {"team": "west_east", "minimum": 30, "maximum": 30}),
        {"op": "any", "children": [score, second]}]}
    if operator == "not":
        score["arguments"].update(minimum=0, maximum=103)
        expression["children"][-1] = {"op": "not", "children": [score]}
    elif operator in {"count", "contradiction"}:
        count = 2 if operator == "contradiction" else 1
        expression["children"][-1].update(op="count", minimum=count, maximum=count)
    elif operator == "correlated":
        expression["children"] = [expression["children"][1], {"op": "any", "children": [
            {"op": "all", "children": [north, score]},
            {"op": "all", "children": [
                _leaf("PRED-DEALER-SOUTH", "auction.dealer", {"seat": "south"}), second]},
        ]}]
    elif operator == "tied":
        score["arguments"].update(minimum=200, maximum=200)
        expression = score
    elif operator == "dealer_not":
        north["arguments"]["seat"] = "east"
        expression = {"op": "not", "children": [north]}
    elif operator == "fixed_conflict":
        expression["children"][-1] = {"op": "all", "children": [score, second]}
    elif operator == "dealer_conflict":
        expression["children"][-1] = _leaf("PRED-DEALER-SOUTH", "auction.dealer", {"seat": "south"})
    elif operator == "completed":
        score["arguments"].update(minimum=200, maximum=200)
        expression["children"][-1] = score
    scenario = _scenario(constraints=expression,
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "bidding"}, "report"),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}})
    observed = []

    def construct(compiler, *_args, **_kwargs):
        observed.append((compiler.dealer, compiler.match_score))
        return OrToolsCandidateAttemptV1("incomplete", None, "dispatch_only")

    monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__", construct)
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-COMPOSED-INPUTS", bundle_hash="a" * 64).execute(
        scenario, _proof(), maximum_candidates=1, maximum_seconds=5,
        maximum_proof_nodes=1000, maximum_proof_depth=4)
    if operator in {"contradiction", "fixed_conflict", "dealer_conflict", "completed"}:
        assert result.status == "infeasible" and observed == []
        return
    assert result.status == "incomplete" and len(observed) == 1
    dealer, scores = observed[0]
    if operator == "tied":
        assert scores == (200, 200)
    elif operator == "dealer_not":
        assert dealer != Seat.EAST and scores == (0, 0)
    else:
        assert scores[1] == 30
        if operator == "correlated":
            assert (dealer, scores[0]) in {(Seat.NORTH, 101), (Seat.SOUTH, 103)}
        else:
            assert dealer == Seat.NORTH
            assert scores[0] > 103 if operator == "not" else scores[0] in {101, 103}


@pytest.mark.real_clock
@pytest.mark.parametrize("fixed", [False, True])
def test_initial_input_projection_excludes_only_the_selected_tuple(root, fixed, deterministic_native_budget):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-DEALER", "auction.dealer", {"seat": "north"}),
        _leaf("PRED-SCORE-NS", "state.match_score", {
            "team": "north_south", "minimum": 101, "maximum": 101 if fixed else 102}),
        _leaf("PRED-SCORE-WE", "state.match_score", {"team": "west_east", "minimum": 30, "maximum": 30}),
    ]}, outcome=_leaf("PRED-REPORT", "state.phase", {"state": "bidding"}, "report"))
    arguments = dict(rules=rules, deadline=time.monotonic() + 5, maximum_conflicts=1000)
    dealer, scores, _partial = executor_module._required_search_initial_inputs_v3(scenario, **arguments)
    assert (dealer, scores) == (Seat.NORTH, (101, 30))
    excluded = [(int(dealer), *scores)]
    dealer, scores, _partial = executor_module._required_search_initial_inputs_v3(
        scenario, excluded_inputs=excluded, **arguments)
    if fixed:
        assert scores is None
    else:
        assert (dealer, scores) == (Seat.NORTH, (102, 30))
        excluded.append((int(dealer), *scores))
        assert executor_module._required_search_initial_inputs_v3(
            scenario, excluded_inputs=excluded, **arguments)[1] is None


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["second", "exhausted", "unknown", "cancelled", "deadline"])
def test_generic_search_retries_only_infeasible_initial_tuples(root, monkeypatch, stop, deterministic_native_budget):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-SCORE-NS", "state.match_score", {"team": "north_south", "minimum": 101, "maximum": 102}),
        _leaf("PRED-SCORE-WE", "state.match_score", {"team": "west_east", "minimum": 30, "maximum": 30}),
    ]}, outcome=_leaf("PRED-REPORT", "state.phase", {"state": "bidding"}, "report"),
        decision_point={"kind": "bid", "selector": {"actionIndex": 0}})
    observed = []
    cancelled = [False]
    clock = deterministic_native_budget

    def construct(compiler, _scenario, excluded, **options):
        observed.append((compiler.match_score, options["maximum_seconds"]))
        clock[0] += 0.25
        assert excluded == ()
        if len(observed) == 1:
            if stop == "unknown":
                monkeypatch.setattr(cp_model.CpSolver, "solve", lambda *_args: cp_model.UNKNOWN)
            cancelled[0] = stop == "cancelled"
            if stop == "deadline":
                clock[0] += 10
            return OrToolsCandidateAttemptV1("infeasible", None, "tuple_has_no_round")
        return OrToolsCandidateAttemptV1("infeasible" if stop == "exhausted" else "incomplete", None, "dispatch_only")

    monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__", construct)
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-INPUT-RETRIES", bundle_hash="a" * 64).execute(
        scenario, _proof(), maximum_candidates=1, maximum_seconds=5,
        maximum_stage_seconds=3, maximum_proof_nodes=1000, maximum_proof_depth=4,
        cancel_check=lambda: cancelled[0])
    assert result.status == ("not_found" if stop == "exhausted" else "incomplete")
    assert [scores for scores, _seconds in observed] == (
        [(101, 30), (102, 30)] if stop in {"second", "exhausted"} else [(101, 30)])
    if len(observed) == 2:
        assert 0 < observed[1][1] < observed[0][1] <= 3
    if stop in {"unknown", "cancelled", "deadline"}:
        receipt = result.execution_receipt["incompleteCompilation"]
        assert receipt["exhaustionReason"] == {
            "unknown": "expression_projection_incomplete", "cancelled": "cancelled", "deadline": "maximum_seconds",
        }[stop]


@pytest.mark.real_clock
@pytest.mark.parametrize("threshold,hardness,decision_kind", [
    (100, "hard", "play"), (73, "hard", "play"),
    (100, "prefer", "play"), (100, "report", "play"), (100, "report", "escalation"),
])
def test_initial_score_proposals_follow_config_without_qualifying_escalation(
    root, threshold, hardness, decision_kind, deterministic_native_budget
):
    data = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml").data
    data["escalation"]["sun_score_threshold"] = threshold
    rules = RuleBook(data)
    scenario = _scenario(constraints=_leaf("PRED-DOUBLE", "escalation.level", {"level": "double"}, hardness),
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": decision_kind, "selector": {"actionIndex": 0} if decision_kind == "escalation"
            else {"trickIndex": 6}})
    excluded, observed = [], []
    for _ in range(5):
        dealer, scores, partial = executor_module._required_search_initial_inputs_v3(scenario,
            rules=rules, deadline=time.monotonic() + 5, maximum_conflicts=1000, excluded_inputs=excluded)
        if scores is None:
            break
        assert partial is True
        observed.append(scores)
        excluded.append((int(dealer), *scores))
    expected = {(0, 0), (0, threshold + 1), (threshold + 1, 0), (threshold + 1, threshold + 1)}
    assert observed[0] == (0, 0)
    assert set(observed) == (expected if hardness == "hard" or decision_kind == "escalation" else {(0, 0)})
    assert scores is None


@pytest.mark.real_clock
def test_initial_escalation_score_proposals_preserve_authored_high_score_ties(root, deterministic_native_budget):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-SCORE", "state.match_score", {"team": "north_south", "minimum": 200, "maximum": 200}),
        _leaf("PRED-NORMAL", "escalation.level", {"level": "normal"}),
    ]}, outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"))
    _dealer, scores, partial = executor_module._required_search_initial_inputs_v3(scenario,
        rules=rules, deadline=time.monotonic() + 5, maximum_conflicts=1000)
    assert scores == (200, 200) and partial is True


@pytest.mark.real_clock
@pytest.mark.parametrize("score_scope", ["omitted", "one_fixed", "both_fixed"])
def test_native_default_score_exhaustion_is_not_global_infeasibility(
    root, monkeypatch, score_scope, deterministic_native_budget
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    constraints = [
        _leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
        _leaf("PRED-DOUBLE", "escalation.level", {"level": "double"}),
    ]
    for team in (() if score_scope == "omitted" else ("north_south",) if score_scope == "one_fixed"
                 else ("north_south", "west_east")):
        constraints.append(_leaf(f"PRED-SCORE-{team.upper().replace('_', '-')}", "state.match_score", {
            "team": team, "minimum": 0, "maximum": 0}))
    scenario = _scenario(constraints={"op": "all", "children": constraints},
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "final_dealt"}, "report"),
        decision_point={"kind": "escalation", "selector": {"actingSeat": "west", "actionIndex": 0}})
    if score_scope != "both_fixed":
        monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__",
            lambda *_args, **_kwargs: OrToolsCandidateAttemptV1("infeasible", None, "no_witness"))
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-DEFAULT-SCORE-SCOPE", bundle_hash="a" * 64).execute(
        scenario, _proof(), maximum_candidates=1, maximum_seconds=5, maximum_stage_seconds=3)
    assert result.status == ("infeasible" if score_scope == "both_fixed" else "not_found"), result.execution_receipt


@pytest.mark.real_clock
@pytest.mark.parametrize("score_scope", ["range", "omitted", "one_fixed"])
def test_native_search_retries_score_until_sun_escalation_is_reachable(
    root, monkeypatch, score_scope, deterministic_native_budget
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    constraints = [
        _leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
    ]
    if score_scope == "range":
        constraints.extend([
        _leaf("PRED-SCORE-NS", "state.match_score", {"team": "north_south", "minimum": 100, "maximum": 101}),
        _leaf("PRED-SCORE-WE", "state.match_score", {"team": "west_east", "minimum": 30, "maximum": 30}),
        ])
    elif score_scope == "one_fixed":
        constraints.append(_leaf("PRED-SCORE-NS", "state.match_score", {
            "team": "north_south", "minimum": 0, "maximum": 0}))
    scenario = _scenario(constraints={"op": "all", "children": constraints},
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "final_dealt"}, "report"),
        decision_point={"kind": "escalation", "selector": {"actingSeat": "west", "actionIndex": 0}})
    proof = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "complete_state", "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": -100000,
        "budgetProfile": "test_bounded", "horizon": {"unit": "plies", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    observed = []
    original = GenericCompleteTraceCompilerV1.__call__

    def observe(compiler, *args, **kwargs):
        result = original(compiler, *args, **kwargs)
        observed.append((compiler.match_score, result.status))
        return result

    monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__", observe)
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-REACHABLE-SUN-ESCALATION", bundle_hash="a" * 64).execute(
        scenario, proof, maximum_candidates=1, maximum_seconds=30, maximum_stage_seconds=15,
        maximum_proof_nodes=1000, maximum_proof_depth=4)
    assert result.status == "qualified", json.dumps({
        "execution": result.execution_receipt,
        "proof": result.proof,
        "evaluation": result.predicate_evaluation.to_dict() if result.predicate_evaluation else None,
    }, sort_keys=True)
    expected = [(100, 30), (101, 30)] if score_scope == "range" else [(0, 0), observed[-1][0]]
    if score_scope != "range":
        assert sorted(expected[1]) == [0, 101]
        if score_scope == "one_fixed":
            assert expected[1][0] == 0
    assert observed == [(expected[0], "infeasible"), (expected[1], "feasible")]
    assert result.candidate.trace["matchScore"] == list(expected[1])
    assert result.candidate.trace["play"] is None
    assert len(result.candidate.trace["constructionPlay"]["plays"]) == 32
    assert result.proof["completionStatus"] == "bounded_complete"


@pytest.mark.parametrize("tail_first", [False, True])
@pytest.mark.parametrize("quality_mode,bid_boundary,bid_unit", [
    (None, None, None),
    ("exact", None, None),
    ("budgeted", "current_raw_point_margin_v1", "plies"),
    ("budgeted", "auction_hand_margin_v1", "plies"),
    ("budgeted", "auction_hand_margin_v1", "decisions"),
])
def test_exact_quality_search_does_not_hard_require_heuristic_bids(
    root, monkeypatch, tail_first, quality_mode, bid_boundary, bid_unit,
):
    from baloot_platform.research.ortools_tail_first_candidate import (
        OrToolsTailFirstCandidateCompilerV1,
    )

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    quality_settings = {}
    if quality_mode is not None:
        profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
            encoding="utf-8"))["scenarioResearchV4"]
        decisions = {kind: {"utility": "awarded_point_margin", "budgetProfile":
            "offline_heavyweight_v1" if quality_mode == "exact" else "bounded_v1"}
            for kind in SCENARIO_DECISION_KINDS}
        if quality_mode == "budgeted":
            for decision in decisions.values():
                decision.update(horizon={"unit": "plies", "value": 1},
                    boundaryEvaluator="current_raw_point_margin_v1")
            decisions["bid"].update(
                horizon={"unit": bid_unit, "value": 1}, boundaryEvaluator=bid_boundary,
            )
        quality_settings = dict(
            quality_policy=ScenarioPlayingQualityPolicyV1.build({"schemaVersion": 2,
                "mode": quality_mode, "visibility": "complete_state", "decisions": decisions}),
            quality_execution=ScenarioQualityExecutionBudgetV1.build(profile["qualityExecution"]),
            quality_proof_profiles=profile["proofBudgets"]["profiles"],
        )
    constraints = _leaf("PRED-SUN", "auction.contract", {"contract": "sun"})
    if tail_first:
        constraints = {"op": "all", "children": [constraints,
            _leaf("PRED-TAIL-ACE", "play.action_matches", {
                "trickIndex": 5, "position": 3, "selector": {"ranks": ["ace"]},
            })]}
    scenario = _scenario(
        constraints=constraints,
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 5 if tail_first else 0, "position": 3 if tail_first else 0,
            "remainingCards": 3 if tail_first else 8,
            "actingSeat": "west",
        }},
    )
    seen = []

    def construct(compiler, *_args, **_kwargs):
        is_tail = isinstance(compiler, OrToolsTailFirstCandidateCompilerV1)
        seen.append((is_tail, compiler.require_optimal_auction_actions if is_tail
            else compiler.optimal_auction_quality))
        return OrToolsCandidateAttemptV1("incomplete", None, "test_dispatch_only")

    monkeypatch.setattr(GenericCompleteTraceCompilerV1, "__call__", construct)
    monkeypatch.setattr(OrToolsTailFirstCandidateCompilerV1, "__call__", construct)
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-EXACT-BID-UNIVERSE", bundle_hash="a" * 64, **quality_settings)
    result = executor.execute(scenario, _proof(), maximum_candidates=1,
        maximum_seconds=5, maximum_proof_nodes=1000, maximum_proof_depth=12,
        maximum_prefix_lift_attempts=1)
    assert result.status == "incomplete"
    assert seen == [(tail_first, bid_boundary == "auction_hand_margin_v1")]


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("solver_status", ["infeasible", "incomplete"])
def test_auction_policy_exhaustion_does_not_prove_scenario_impossible(rules, monkeypatch, policy, solver_status):
    from types import SimpleNamespace
    from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
    monkeypatch.setattr(OrToolsScenarioModelV1, "solve",
        lambda *args, **kwargs: SimpleNamespace(status=solver_status))
    scenario = _scenario(constraints=_leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-REPORT", "outcome.team_round",
            {"team": "north_south", "status": "won"}, "report"))
    attempt = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True,
        optimal_auction_quality=policy)(scenario, (), seed=0, maximum_seconds=15)
    assert attempt.status == ("not_found" if policy and solver_status == "infeasible" else solver_status)
    if policy and solver_status == "infeasible":
        assert attempt.reason == "auction_policy_no_witness"


@pytest.mark.real_clock
def test_complete_trace_compiler_shares_one_deadline_across_stages(
    rules, monkeypatch
):
    clock = [0.0]
    escalation_allowances = []
    deal_allowances = []

    monkeypatch.setattr(
        executor_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]),
    )

    def solve_escalation(_stage, **kwargs):
        escalation_allowances.append(kwargs["maximum_seconds"])
        clock[0] += 0.15
        return SimpleNamespace(status="satisfied")

    def solve_deal(_model, **kwargs):
        deal_allowances.append(kwargs["maximum_seconds"])
        clock[0] += 0.5
        return SimpleNamespace(status="infeasible")

    monkeypatch.setattr(executor_module.OrToolsEscalationStageV1, "solve", solve_escalation)
    monkeypatch.setattr(executor_module.OrToolsScenarioModelV1, "solve", solve_deal)
    scenario = _scenario(
        constraints=_leaf(
            "PRED-DEADLINE-HOKUM",
            "auction.contract",
            {"contract": "hokum"},
        ),
        decision_point={
            "kind": "escalation",
            "selector": {"actingSeat": "west", "actionIndex": 0},
        },
        outcome=_leaf(
            "PRED-DEADLINE-REPORT",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )

    attempt = GenericCompleteTraceCompilerV1(
        rules=rules,
        native_cp_sat_scoring=True,
    )(scenario, (), seed=0, maximum_seconds=1.0)

    assert escalation_allowances == pytest.approx([1.0, 0.85, 0.7, 0.55])
    assert deal_allowances == pytest.approx([0.4])
    assert attempt == OrToolsCandidateAttemptV1(
        "incomplete", None, "deal_auction_budget"
    )


def test_escalation_search_rejects_structurally_forced_sun_root(
    rules, monkeypatch
):
    allowed_pairs = []
    original_add_allowed_assignments = cp_model.CpModel.add_allowed_assignments

    def track_allowed_assignments(model, expressions, tuples_list):
        if len(expressions) == 2:
            rows = tuple(tuple(row) for row in tuples_list)
            allowed_pairs.append((tuple(item.name for item in expressions), rows))
            return original_add_allowed_assignments(model, expressions, rows)
        return original_add_allowed_assignments(model, expressions, tuples_list)

    def forbidden(*args, **kwargs):
        pytest.fail("native construction used Python escalation legality")

    monkeypatch.setattr(
        cp_model.CpModel, "add_allowed_assignments", track_allowed_assignments
    )
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    scenario = _scenario(
        constraints=_leaf(
            "PRED-ESCALATION-FORCED-SUN",
            "auction.contract",
            {"contract": "sun"},
        ),
        decision_point={
            "kind": "escalation",
            "selector": {"actingSeat": "west", "actionIndex": 0},
        },
        outcome=_leaf(
            "PRED-ESCALATION-FORCED-REPORT",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )

    attempt = GenericCompleteTraceCompilerV1(
        rules=rules,
        match_score=(0, 0),
        native_cp_sat_scoring=True,
    )(scenario, (), seed=0, maximum_seconds=5)

    assert attempt == OrToolsCandidateAttemptV1(
        "infeasible",
        None,
        "escalation_decision_structurally_infeasible",
    )

    hokum = _scenario(
        constraints=_leaf(
            "PRED-ESCALATION-HOKUM",
            "auction.contract",
            {"contract": "hokum"},
        ),
        decision_point={
            "kind": "escalation",
            "selector": {"actingSeat": "west", "actionIndex": 0},
        },
        outcome=_leaf(
            "PRED-ESCALATION-HOKUM-REPORT",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )
    positive = GenericCompleteTraceCompilerV1(
        rules=rules,
        match_score=(0, 0),
        native_cp_sat_scoring=True,
    )(hokum, (), seed=0, maximum_seconds=15)

    assert positive.status == "feasible", positive
    assert positive.candidate is not None
    assert positive.candidate.trace["contract"] == "hokum"
    strategic_tables = [
        names
        for names, rows in allowed_pairs
        if rows == ((0, 1), (2, 1))
    ]
    assert any("declarer" in names[0] for names in strategic_tables)
    assert not any("buyer" in names[0] for names in strategic_tables)


@pytest.mark.parametrize("native", [False, True], ids=["legacy", "native-v3"])
def test_search_construction_can_require_a_native_ekka_action(rules, monkeypatch, native):
    if native:
        import baloot_platform.research.ortools_scenario_model as auction_model

        def forbidden(*args, **kwargs):
            pytest.fail("native Ekka construction used legacy auction enumeration")

        monkeypatch.setattr(auction_model, "_enumerate_auction", forbidden)
    else:
        original_solve = OrToolsPlayTailStageV1.solve

        def feasible_witness(self, **kwargs):
            return replace(original_solve(self, **kwargs), solver_status="feasible")

        monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", feasible_witness)
    scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf(
                    "PRED-EKKA-HOKUM",
                    "auction.contract",
                    {"contract": "hokum"},
                ),
                _leaf(
                    "PRED-EKKA-TRUMP",
                    "auction.trump",
                    {"suit": "hearts"},
                ),
                _leaf(
                    "PRED-EKKA-CARD",
                    "deal.card_owner",
                    {"card": 31, "seat": "north", "handStage": "post_purchase"},
                ),
                _leaf(
                    "PRED-EKKA-ACTION",
                    "play.action_matches",
                    {
                        "trickIndex": 0,
                        "position": 0,
                        "selector": {"ekka": "required"},
                    },
                ),
            ],
        },
        outcome=_leaf(
            "PRED-EKKA-OUTCOME-REPORT",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )

    attempt = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=native)(
        scenario,
        (),
        seed=0,
        maximum_seconds=60,
    )

    assert attempt.status == "feasible", attempt.partial_receipt or attempt
    assert attempt.candidate is not None
    assert attempt.candidate.trace["contract"] == "hokum"
    assert attempt.candidate.trace["play"]["plays"][0]["ekka"] is True


@pytest.mark.parametrize("native", [False, True])
def test_bid_construction_binds_nonterminal_auction_prefix(rules, monkeypatch, native):
    if native:
        import baloot_platform.research.ortools_scenario_model as auction_model

        def forbidden(*args, **kwargs):
            pytest.fail("native construction used legacy auction enumeration")

        monkeypatch.setattr(auction_model, "_enumerate_auction", forbidden)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-AUCTION-PREFIX-OPEN",
                "auction.terminal",
                {"terminal": False},
            ),
            "decisionPoint": {
                "kind": "bid",
                "selector": {"actionIndex": 4},
            },
            "outcome": _leaf(
                "PRED-AUCTION-PREFIX-REPORT",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
                "report",
            ),
        }
    )

    attempt = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=native)(
        scenario,
        (),
        seed=0,
        maximum_seconds=15,
    )

    assert attempt.status == "feasible"
    assert attempt.candidate is not None
    assert attempt.candidate.trace["auctionPredicateView"] == "prefix"
    auction = attempt.candidate.trace["auction"]
    assert len(auction["actions"]) == 4
    assert auction["prefixTerminal"] is False
    assert auction["legalNextActions"]
    bindings = DefaultStageContextCompilerV1(rules, native_cp_sat_authority=native)(attempt.candidate, None)
    evaluation = OrToolsResearchCompilerV1().evaluate(
        scenario,
        attempt.candidate,
        bindings,
        maximum_seconds=5,
    )
    assert evaluation.status == "satisfied"
    observed = {row["predicateId"]: row for row in evaluation.evidence}
    assert observed["PRED-AUCTION-PREFIX-OPEN"]["observed"] == {"terminal": False}


def _matching_scenario(candidate: OrToolsNeutralCandidateV1) -> ResearchScenarioV1:
    trace = candidate.trace
    score = trace["score"]
    purchase_suit = trace["auction"]["completion"]["trumpSuit"]
    # Purchase suit is independent of Sun trump and is bound in the deal receipt.
    purchase_suit = candidate.trace["play"]["initialHands"]
    del purchase_suit
    deck_purchase = candidate.trace["deckOrder"][20]
    suit = Suit(deck_purchase // 8).name.lower()
    winner = score["roundWinner"]
    return _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf("PRED-DEAL-TRUE", "deal.purchase_suit", {"suit": suit}),
                _leaf(
                    "PRED-AUCTION-TRUE",
                    "auction.contract",
                    {"contract": trace["contract"]},
                ),
                _leaf(
                    "PRED-ESCALATION-TRUE",
                    "escalation.level",
                    {"level": trace["escalation"]["observation"]["level"]},
                ),
                {
                    "op": "any",
                    "children": [
                        _leaf(
                            "PRED-PROJECT-TRUE",
                            "project.count",
                            {"team": "north_south", "minimum": 0, "maximum": 100},
                        ),
                        _leaf(
                            "PRED-AUCTION-FALSE",
                            "auction.contract",
                            {
                                "contract": "hokum"
                                if trace["contract"] == "sun"
                                else "sun"
                            },
                        ),
                    ],
                },
                {
                    "op": "not",
                    "children": [
                        _leaf(
                            "PRED-DEAL-FALSE",
                            "deal.purchase_suit",
                            {
                                "suit": next(
                                    value
                                    for value in (
                                        "diamonds",
                                        "clubs",
                                        "hearts",
                                        "spades",
                                    )
                                    if value != suit
                                )
                            },
                        )
                    ],
                },
                {
                    "op": "count",
                    "minimum": 1,
                    "maximum": 1,
                    "children": [
                        _leaf(
                            "PRED-PLAY-TRUE",
                            "play.trick_index",
                            {"minimum": 6, "maximum": 6},
                        ),
                        _leaf(
                            "PRED-STATE-FALSE",
                            "state.phase",
                            {"state": "created"},
                        ),
                    ],
                },
            ],
        },
        outcome=_leaf(
            "PRED-OUTCOME-TRUE",
            "outcome.team_round",
            {"team": winner, "status": "won"},
        ),
    )


def test_preflight_starts_at_most_selective_stage_and_rejects_unsafe_not():
    safe = _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf("PRED-DEAL-STAGE", "deal.purchase_suit", {"suit": "spades"}),
                _leaf(
                    "PRED-PROOF-STAGE",
                    "proof.winning_root_action_count",
                    {"minimum": 1, "maximum": 1},
                ),
            ],
        },
        outcome=_leaf(
            "PRED-OUTCOME-STAGE",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
        ),
    )
    compiled = OrToolsResearchCompilerV1().preflight(safe)
    assert compiled.supported
    assert compiled.start_stage == "proof"
    assert compiled.compiled_stages == ("proof", "outcome", "deal")

    unsafe = _scenario(
        constraints={
            "op": "not",
            "children": [
                _leaf(
                    "PRED-UNSAFE-NOT",
                    "state.observer_visibility",
                    {"observer": "north", "classification": "public_fact"},
                    "report",
                )
            ],
        },
        outcome=_leaf(
            "PRED-UNSAFE-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
        ),
    )
    rejected = OrToolsResearchCompilerV1().preflight(unsafe)
    assert not rejected.supported
    assert rejected.unsupported == (
        {"predicateId": "PRED-UNSAFE-NOT", "code": "unsafe_not"},
    )


def test_generic_compiler_builds_complete_trace_and_exact_deck_no_goods(
    rules,
    complete_candidate,
):
    assert len(complete_candidate.trace["play"]["plays"]) == 32
    assert complete_candidate.trace["play"]["decisionState"]["prefixLength"] == 24
    assert sorted(complete_candidate.stage_receipts) == [
        "auction",
        "deal",
        "escalation",
        "outcome",
        "playPrefix",
        "projects",
        "state",
    ]
    scenario = _scenario(
        constraints=_leaf(
            "PRED-NO-GOOD-REPORT-STATE",
            "state.phase",
            {"state": "playing"},
            "report",
        ),
        outcome=_leaf(
            "PRED-NO-GOOD-REPORT-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )
    second = GenericCompleteTraceCompilerV1(rules=rules)(
        scenario,
        (complete_candidate.no_good_vector,),
        seed=0,
        maximum_seconds=15,
    )
    assert second.status == "feasible", second
    assert second.candidate is not None
    assert second.candidate.no_good_vector != complete_candidate.no_good_vector


def test_cross_stage_expression_compiles_all_any_safe_not_and_count(
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)
    bindings = DefaultStageContextCompilerV1(rules)(complete_candidate, None)
    result = OrToolsResearchCompilerV1().evaluate(
        scenario,
        complete_candidate,
        bindings,
        maximum_seconds=15,
    )
    assert result.status == "satisfied"
    assert result.satisfied
    assert len(result.evidence) == 9
    assert {row["kind"].split(".", 1)[0] for row in result.evidence} == {
        "deal",
        "auction",
        "escalation",
        "project",
        "play",
        "state",
        "outcome",
    }


def test_v3_cross_stage_expression_certifies_all_any_safe_not_and_count(
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=compiler.preflight(scenario).compiled_stages,
        native_cp_sat_authority=True,
    )(complete_candidate, None)

    result = compiler.evaluate(
        scenario,
        complete_candidate,
        bindings,
        maximum_seconds=15,
    )

    assert result.status == "satisfied"
    assert result.satisfied
    assert len(result.evidence) == 9


@pytest.mark.real_clock
@pytest.mark.parametrize("native", [False, True])
def test_predicate_budget_is_shared_across_stage_models(
    rules, complete_candidate, monkeypatch, native
):
    scenario = _scenario(
        constraints={"op": "all", "children": [
            _leaf("PRED-BUDGET-DEAL", "deal.purchase_card", {
                "card": complete_candidate.trace["deckOrder"][20],
            }),
            _leaf("PRED-BUDGET-AUCTION", "auction.contract", {
                "contract": complete_candidate.trace["contract"],
            }),
        ]},
        outcome=_leaf("PRED-BUDGET-REPORT", "state.phase", {
            "state": "playing",
        }, "report"),
    )
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=native)
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=compiler.preflight(scenario).compiled_stages,
        native_cp_sat_authority=native,
    )(complete_candidate, None)
    assert len({id(binding.context.model) for binding in bindings.values()}) > 1
    solve = executor_module._solve_cp_sat_with_cancel_v1
    clock = [0.0]
    calls = []

    def consume_budget(solver, model, cancel_check):
        status = solve(solver, model, cancel_check)
        calls.append(solver.parameters.max_time_in_seconds)
        if len(calls) == 1:
            assert status == cp_model.OPTIMAL
            clock[0] = 3.0
        return status

    monkeypatch.setattr(executor_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(executor_module, "_solve_cp_sat_with_cancel_v1", consume_budget)
    result = compiler.evaluate(
        scenario, complete_candidate, bindings, maximum_seconds=2.0
    )

    assert result.status == ("incomplete" if native else "satisfied")
    if native:
        assert len(calls) == 1
    else:
        assert len(calls) > 1


def test_v3_generated_auction_outcomes_still_require_native_qualification(root, monkeypatch):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    raw.update(escalationActions=[], projectDeclarations=[], projectAssertions=[],
               playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"}, maximum_seconds=15)
    owner = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-AUCTION-QUALIFICATION", bundle_hash="a" * 64)
    complete_candidate = owner._candidate_from_normalized_exact(normalized, prefix, origin="search")
    monkeypatch.setattr(executor_module, "OrToolsEscalationStageV1",
                        lambda *args, **kwargs: pytest.fail("auction-only qualification compiled escalation"))
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    contract = complete_candidate.trace["contract"]
    buyer = complete_candidate.trace["buyer"]
    for kind, arguments, satisfied in (
        ("auction.contract", {"contract": contract}, True),
        ("auction.contract", {"contract": "hokum" if contract == "sun" else "sun"}, False),
        ("auction.buyer", {"seat": buyer}, True),
        ("auction.buyer", {"seat": Seat.parse(buyer).next().name.lower()}, False),
    ):
        scenario = _scenario(
            constraints=_leaf("PRED-AUCTION-REQUIRED", kind, arguments),
            outcome=_leaf("PRED-CONTRACT-REPORT", "auction.contract", {"contract": contract}, "report"))
        bindings = DefaultStageContextCompilerV1(rules,
            required_stages=compiler.preflight(scenario).compiled_stages,
            native_cp_sat_authority=True)(complete_candidate, None)
        result = compiler.evaluate(scenario, complete_candidate, bindings, maximum_seconds=15)
        assert result.satisfied is satisfied, (kind, arguments, result)
        assert result.status == ("satisfied" if satisfied else "target_miss")
        assert set(bindings) == {"auction"}
        substituted = dict(bindings)
        substituted["auction"] = replace(bindings["auction"], source_evidence_hash="f" * 64)
        with pytest.raises(ValueError, match="does not own candidate evidence"):
            compiler.evaluate(scenario, complete_candidate, substituted, maximum_seconds=15)


def test_v3_cross_stage_expression_exhausts_false_any(rules, complete_candidate):
    purchase_card = complete_candidate.trace["deckOrder"][20]
    actual_suit = Suit(purchase_card // 8)
    other_suit = next(suit for suit in Suit if suit != actual_suit)
    winner = complete_candidate.trace["score"]["roundWinner"]
    loser = Team[winner.upper()].other.name.lower()
    scenario = _scenario(
        constraints={
            "op": "any",
            "children": [
                _leaf(
                    "PRED-V3-FALSE-DEAL",
                    "deal.purchase_suit",
                    {"suit": other_suit.name.lower()},
                ),
                _leaf(
                    "PRED-V3-FALSE-OUTCOME",
                    "outcome.team_round",
                    {"team": loser, "status": "won"},
                ),
            ],
        },
        outcome=_leaf(
            "PRED-V3-FALSE-REPORT",
            "outcome.team_round",
            {"team": winner, "status": "won"},
            "report",
        ),
    )
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=compiler.preflight(scenario).compiled_stages,
        native_cp_sat_authority=True,
    )(complete_candidate, None)

    result = compiler.evaluate(
        scenario,
        complete_candidate,
        bindings,
        maximum_seconds=15,
    )

    assert result.status == "target_miss"
    assert not result.satisfied


def test_v3_cross_stage_conjunction_uses_native_stage_literals(
    rules,
    complete_candidate,
):
    score = complete_candidate.trace["score"]
    purchase_card = complete_candidate.trace["deckOrder"][20]
    scenario = _scenario(
        constraints=_leaf(
            "PRED-V3-DEAL",
            "deal.purchase_suit",
            {"suit": Suit(purchase_card // 8).name.lower()},
        ),
        outcome=_leaf(
            "PRED-V3-OUTCOME",
            "outcome.team_round",
            {"team": score["roundWinner"], "status": "won"},
        ),
    )
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    preflight = compiler.preflight(scenario)
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=preflight.compiled_stages,
        native_cp_sat_authority=True,
    )(complete_candidate, None)

    result = compiler.evaluate(
        scenario,
        complete_candidate,
        bindings,
        maximum_seconds=15,
    )

    assert result.status == "satisfied"
    assert result.satisfied
    assert {row["predicateId"] for row in result.evidence} == {
        "PRED-V3-DEAL",
        "PRED-V3-OUTCOME",
    }


def test_v3_play_and_state_share_native_transition_variables(
    rules,
    complete_candidate,
):
    scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf(
                    "PRED-V3-PLAY-TRICK",
                    "play.trick_index",
                    {"minimum": 6, "maximum": 6},
                ),
                _leaf(
                    "PRED-V3-PLAYING",
                    "state.phase",
                    {"state": "playing"},
                ),
            ],
        },
        outcome=_leaf(
            "PRED-V3-OUTCOME-REPORT",
            "outcome.team_round",
            {
                "team": complete_candidate.trace["score"]["roundWinner"],
                "status": "won",
            },
            "report",
        ),
    )
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    preflight = compiler.preflight(scenario)
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=preflight.compiled_stages,
        native_cp_sat_authority=True,
    )(complete_candidate, None)

    play_context = bindings["playPrefix"].context
    state_context = bindings["state"].context
    assert play_context is state_context
    card = play_context.play_slots[0].card
    assert card.model_proto is state_context.model.proto
    assert card.Index() == play_context.native_build._extract.card_ids[0].Index()
    expected_card = Card.parse(complete_candidate.trace["play"]["plays"][0]["card"]).id
    assert list(card.proto.domain) == [expected_card, expected_card]
    result = compiler.evaluate(
        scenario,
        complete_candidate,
        bindings,
        maximum_seconds=15,
    )

    assert result.status == "satisfied"
    assert result.satisfied


@pytest.mark.parametrize("operator", ["any", "count"])
def test_soft_predicate_cannot_satisfy_hard_any_or_count(
    rules,
    complete_candidate,
    operator,
):
    purchase_card = complete_candidate.trace["deckOrder"][20]
    purchase_suit = Suit(purchase_card // 8).name.lower()
    other_suit = next(
        item.name.lower() for item in Suit if item.name.lower() != purchase_suit
    )
    children = [
        _leaf("PRED-HARD-FALSE", "deal.purchase_suit", {"suit": other_suit}),
        _leaf(
            "PRED-SOFT-TRUE",
            "deal.purchase_suit",
            {"suit": purchase_suit},
            "prefer" if operator == "any" else "report",
        ),
    ]
    constraints = {"op": operator, "children": children}
    if operator == "count":
        constraints.update({"minimum": 1, "maximum": 1})
    scenario = _scenario(
        constraints=constraints,
        outcome=_leaf(
            "PRED-HARD-OUTCOME",
            "outcome.team_round",
            {
                "team": complete_candidate.trace["score"]["roundWinner"],
                "status": "won",
            },
        ),
    )

    result = OrToolsResearchCompilerV1().evaluate(
        scenario,
        complete_candidate,
        DefaultStageContextCompilerV1(rules)(complete_candidate, None),
        maximum_seconds=15,
    )

    assert result.status == "target_miss"
    assert not result.satisfied
    assert len(result.evidence) == 3


@pytest.mark.parametrize("hardness", ["report", "prefer"])
@pytest.mark.parametrize("soft_first", [True, False])
@pytest.mark.parametrize("operator", ["any", "all", "count"])
def test_native_mixed_hardness_qualification_uses_the_hard_model(
    hardness, soft_first, operator,
):
    from baloot_platform.research.scenario_contracts import ScenarioPredicateV1
    from baloot_platform.research.scenario_predicate_lowering_v2 import (
        CpSatActionVariablesV1, CpSatPredicateModelContextV1, CpSatTrickVariablesV1,
    )
    from baloot_platform.research.scenario_project_outcome_lowering_v2 import (
        CpSatOutcomePredicateContextV1,
    )

    soft_model = cp_model.CpModel()
    # Distinct models may reuse indices; these are not the hard-model variables.
    for index in range(32):
        soft_model.new_bool_var(f"unrelated_{index}")
    zero, one, absent = (soft_model.new_constant(value) for value in (0, 1, 2))
    soft_context = CpSatOutcomePredicateContextV1(
        soft_model, "soft-outcome", "scoring", "a" * 64,
        zero, zero, absent, (zero, one), (zero, one), (zero, zero),
        (zero, zero), zero, zero, 152,
    )
    hard_model = cp_model.CpModel()
    actions = tuple(CpSatActionVariablesV1(
        f"ACTION-{index}", hard_model.new_constant(index),
        hard_model.new_constant(index), hard_model.new_constant(index),
        hard_model.new_constant(0), hard_model.new_constant(1),
    ) for index in range(4))
    hard_context = CpSatPredicateModelContextV1(
        hard_model, "hard-play", {0: CpSatTrickVariablesV1(0, actions, actions[0])},
        (), hard_model.new_constant(1),
    )
    hard = _leaf("PRED-HARD-FALSE", "play.action_matches", {
        "trickIndex": 0, "position": 0, "selector": {"cards": [31]},
    })
    soft = _leaf("PRED-SOFT-TRUE", "outcome.team_round", {
        "team": "north_south", "status": "won",
    }, hardness)
    expression = {"op": operator, "children": [soft, hard] if soft_first else [hard, soft]}
    if operator == "count":
        expression.update(minimum=1, maximum=1)
    scenario = _scenario(constraints=expression,
        outcome={**soft, "predicateId": "PRED-SOFT-OUTCOME"})
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    lowering = compiler.manifest.lower_predicate(ScenarioPredicateV1.build(hard), hard_context)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 1
    assert solver.solve(hard_model) == cp_model.OPTIMAL
    assert lowering.extract(solver)["satisfied"] is False
    receipts = {stage: OrToolsResearchStageReceiptV1(
        stage, "b" * 64, "c" * 64, "d" * 64, "OPTIMAL",
    ) for stage in ("playPrefix", "outcome")}
    candidate = OrToolsNeutralCandidateV1("search", {}, {}, (), receipts)
    bindings = {stage: OrToolsStageContextBindingV1(stage, receipts[stage].evidence_hash, context)
                for stage, context in (("playPrefix", hard_context), ("outcome", soft_context))}

    result = compiler.evaluate(scenario, candidate, bindings, maximum_seconds=2)

    assert result.status == "target_miss"
    assert result.satisfied is False


def test_soft_predicate_under_not_does_not_block_hard_qualification(
    rules,
    complete_candidate,
):
    purchase_card = complete_candidate.trace["deckOrder"][20]
    purchase_suit = Suit(purchase_card // 8).name.lower()
    scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf(
                    "PRED-HARD-TRUE",
                    "deal.purchase_suit",
                    {"suit": purchase_suit},
                ),
                {
                    "op": "not",
                    "children": [
                        _leaf(
                            "PRED-SOFT-TRUE-NOT",
                            "deal.purchase_suit",
                            {"suit": purchase_suit},
                            "report",
                        )
                    ],
                },
            ],
        },
        outcome=_leaf(
            "PRED-HARD-OUTCOME-NOT",
            "outcome.team_round",
            {
                "team": complete_candidate.trace["score"]["roundWinner"],
                "status": "won",
            },
        ),
    )

    result = OrToolsResearchCompilerV1().evaluate(
        scenario,
        complete_candidate,
        DefaultStageContextCompilerV1(rules)(complete_candidate, None),
        maximum_seconds=15,
    )

    assert result.status == "satisfied"
    assert result.satisfied
    assert len(result.evidence) == 3


def test_candidate_binding_substitution_is_rejected(rules, complete_candidate):
    scenario = _matching_scenario(complete_candidate)
    bindings = dict(DefaultStageContextCompilerV1(rules)(complete_candidate, None))
    original = bindings["deal"]
    bindings["deal"] = OrToolsStageContextBindingV1("deal", "f" * 64, original.context)
    with pytest.raises(ValueError, match="does not own candidate evidence"):
        OrToolsResearchCompilerV1().evaluate(
            scenario, complete_candidate, bindings, maximum_seconds=15
        )


def test_search_trace_binds_observer_visibility_without_runtime_state(
    rules, complete_candidate
):
    observer = complete_candidate.trace["play"]["decisionState"]["nextActor"]
    scenario = _scenario(
        constraints=_leaf(
            "PRED-OBSERVER-BINDING",
            "state.observer_visibility",
            {"observer": observer, "classification": "public_fact"},
        ),
        outcome=_leaf(
            "PRED-OBSERVER-OUTCOME",
            "outcome.team_round",
            {
                "team": complete_candidate.trace["score"]["roundWinner"],
                "status": "won",
            },
        ),
    )
    result = OrToolsResearchCompilerV1().evaluate(
        scenario,
        complete_candidate,
        DefaultStageContextCompilerV1(rules)(complete_candidate, None),
        maximum_seconds=15,
    )
    assert result.status == "satisfied"
    assert {item["kind"] for item in result.evidence} == {
        "state.observer_visibility",
        "outcome.team_round",
    }


@pytest.mark.parametrize("native", [False, True])
def test_state_only_binding_skips_unrequested_lifecycle_models(
    monkeypatch,
    rules,
    complete_candidate,
    native,
):
    actor = complete_candidate.trace["play"]["decisionState"]["nextActor"]
    scenario = _scenario(
        constraints=_leaf(
            "PRED-STATE-ONLY-ACTOR",
            "state.next_actor",
            {"seat": actor},
        ),
        outcome=_leaf(
            "PRED-STATE-ONLY-REPORT",
            "state.next_actor",
            {"seat": actor},
            "report",
        ),
    )

    def unexpected_project_lifecycle(*_args, **_kwargs):
        raise AssertionError("state-only binding built an unrequested project model")

    if native:
        def unexpected_auction(*_args, **_kwargs):
            raise AssertionError("state-only binding built an unrequested auction model")

        monkeypatch.setattr(
            "baloot_platform.research.ortools_research_executor."
            "OrToolsScenarioModelV1.add_native_auction_fragment", unexpected_auction,
        )
    else:
        monkeypatch.setattr(
            "baloot_platform.research.ortools_research_executor."
            "attach_project_lifecycle_fragment_v1",
            unexpected_project_lifecycle,
        )
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages={"state"},
        native_cp_sat_authority=native,
    )(complete_candidate, None)
    result = OrToolsResearchCompilerV1(native_cp_sat_authority=native).evaluate(
        scenario,
        complete_candidate,
        bindings,
        maximum_seconds=15,
    )

    assert set(bindings) == {"state"}
    assert result.status == "satisfied"
    assert result.satisfied is True


def _two_action_graph(
    *, second_wins: bool = False, bounded: bool = False
) -> ProofGraphV1:
    return ProofGraphV1.build(
        scope="bounded" if bounded else "exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                Team.NORTH_SOUTH.name.lower(),
                (
                    ProofGraphEdgeV1.build(0, "D7", "win"),
                    ProofGraphEdgeV1.build(1, "D8", "other"),
                ),
                actor_seat="north",
            ),
            ProofGraphNodeV1.leaf("win", 1, kind="boundary" if bounded else "terminal"),
            ProofGraphNodeV1.leaf(
                "other",
                1 if second_wins else -1,
                kind="boundary" if bounded else "terminal",
            ),
        ),
        boundary_evaluator="current_raw_point_margin_v1" if bounded else None,
    )


def test_v3_proof_binding_fails_closed_without_native_root_variables():
    proof = SimpleNamespace(
        root_actions=(
            SimpleNamespace(
                action_id=0,
                action_code="D7",
                satisfies=True,
                value=1,
            ),
        ),
        model_document=lambda: {},
    )
    model = cp_model.CpModel()

    with pytest.raises(ValueError, match="missing root-action variable bindings"):
        _proof_root_variables(proof, model, require_native=True)

    winning, values = _proof_root_variables(proof, model, require_native=False)
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(winning[0]) == 1
    assert solver.value(values[0]) == 1


def _two_action_causal_graph() -> ProofGraphV1:
    return ProofGraphV1.build(
        scope="bounded",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                Team.WEST_EAST.name.lower(),
                (
                    ProofGraphEdgeV1.build(0, "D7", "followup"),
                    ProofGraphEdgeV1.build(1, "D8", "lose"),
                ),
                actor_seat="east",
            ),
            ProofGraphNodeV1.decision(
                "followup",
                Team.NORTH_SOUTH.name.lower(),
                (ProofGraphEdgeV1.build(2, "C7", "win"),),
                actor_seat="north",
            ),
            ProofGraphNodeV1.leaf("win", 1, kind="boundary"),
            ProofGraphNodeV1.leaf("lose", -1, kind="boundary"),
        ),
        boundary_evaluator="current_raw_point_margin_v1",
    )


def test_executor_distinguishes_feasible_qualified_and_proof_failed(
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)

    def candidate_compiler(*_args, **_kwargs):
        return OrToolsCandidateAttemptV1("feasible", complete_candidate)

    executor = _executor(rules)
    feasible = executor.execute(
        scenario,
        _proof(),
        candidate_compiler=candidate_compiler,
        maximum_seconds=15,
    )
    assert feasible.status == "feasible"
    assert feasible.proof is None

    proof_scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                scenario.constraints.to_dict(),
                _leaf(
                    "PRED-PROOF-ROOT-COUNT",
                    "proof.winning_root_action_count",
                    {"minimum": 1, "maximum": 1},
                ),
                _leaf(
                    "PRED-TACTIC-REPORT",
                    "tactic.applicable",
                    {"kind": "only_winning_move", "applicable": False},
                    "report",
                ),
            ],
        },
        outcome=scenario.outcome.to_dict(),
    )
    qualified = executor.execute(
        proof_scenario,
        _bounded_proof(claim="selected_action_is_unique_winner", selected_action="D7"),
        candidate_compiler=candidate_compiler,
        graph_compiler=lambda _candidate, _request: _two_action_graph(bounded=True),
        maximum_seconds=15,
    )
    assert qualified.status == "qualified"
    assert qualified.candidate is not None
    assert qualified.proof["completionStatus"] == "bounded_complete"
    assert qualified.research_authority["qualificationAuthority"] == "cp_sat"
    assert qualified.research_authority["bundleRevisionId"] == "CBR-TEST-ORTOOLS-V4"
    assert (
        qualified.research_authority["proofModelHash"]
        == qualified.proof["identities"]["modelHash"]
    )
    documents = qualified.semantic_artifact_documents()
    hash_fields = {
        "constructionModel": "constructionModelHash",
        "proofModel": "proofModelHash",
        "proofGraph": "graphHash",
        "trace": "traceHash",
        "proofOutcome": "proofOutcomeHash",
    }
    for document_name, authority_field in hash_fields.items():
        assert (
            canonical_sha256(documents[document_name])
            == qualified.research_authority[authority_field]
        )
    documents["trace"]["schemaVersion"] = 99
    assert qualified.semantic_artifact_documents()["trace"]["schemaVersion"] == 1
    with pytest.raises(ValueError, match="trace"):
        OrToolsResearchArtifactPreimagesV1.build(
            authority=qualified.research_authority,
            construction_model=qualified.semantic_artifact_documents()[
                "constructionModel"
            ],
            proof_model=qualified.semantic_artifact_documents()["proofModel"],
            proof_graph=qualified.semantic_artifact_documents()["proofGraph"],
            trace=documents["trace"],
            proof_outcome=qualified.semantic_artifact_documents()["proofOutcome"],
        )
    assert {row["kind"] for row in qualified.predicate_evaluation.evidence} >= {
        "proof.winning_root_action_count",
        "tactic.applicable",
    }
    tactic_evidence = next(
        row
        for row in qualified.predicate_evaluation.evidence
        if row["kind"] == "tactic.applicable"
    )
    assert tactic_evidence["observed"]["qualificationSource"] == (
        "native_cp_sat_root_win_variables"
    )
    assert tactic_evidence["observed"]["winningVariables"] == [
        "root_wins_0",
        "root_wins_1",
    ]
    assert tactic_evidence["observed"]["applicable"] is True
    assert tactic_evidence["observed"]["verified"] is True

    failed = executor.execute(
        scenario,
        _bounded_proof(claim="outcome_forced"),
        candidate_compiler=candidate_compiler,
        graph_compiler=lambda _candidate, _request: _two_action_graph(bounded=True),
        maximum_seconds=15,
    )
    assert failed.status == "proof_failed"
    assert failed.proof["completionStatus"] == "proof_failed"
    assert failed.rejection_hashes == (failed.candidate.sha256,)


@pytest.mark.parametrize("visibility", ["complete_state", "actor_information_set"])
def test_native_exact_proof_cannot_bypass_hard_qualification(root, visibility):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    final_action = prefix["playActions"].pop()
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-NATIVE-QUALIFICATION-TEST", bundle_hash="a" * 64,
    )

    def scenario(count, hardness="hard"):
        return _scenario(
            origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
            decision_point={"kind": "play", "selector": {
                "trickIndex": 7, "actingSeat": final_action["actor"],
            }},
            constraints=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}),
            outcome=_leaf("PRED-ROOT-COUNT", "proof.winning_root_action_count", count, hardness),
        )

    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": visibility,
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": "north_south", "utility": "round_win", "successThreshold": 1,
        "budgetProfile": "test_exact",
    })
    limits = {"maximum_seconds": 15, "maximum_proof_nodes": 1000,
              "maximum_proof_depth": 1, "maximum_hidden_worlds": 2}
    base_scenario = scenario({"minimum": 0, "maximum": 1})
    baseline = executor.execute(base_scenario, request, **limits)
    assert baseline.status == "qualified", baseline.execution_receipt
    assert baseline.research_authority["backendId"] == "ortools_cp_sat_v3"
    assert baseline.research_authority["qualificationAuthority"] == "cp_sat"
    actual_count = baseline.proof["winningRootActionCount"]
    assert actual_count in (0, 1)
    false_count = {"minimum": 1 - actual_count, "maximum": 1 - actual_count}
    unsupported = executor.compiler.preflight(scenario(false_count, "prefer"))
    assert {"predicateId": "PRED-ROOT-COUNT", "code": "unsupported_hardness"} in unsupported.unsupported
    for hardness in ("hard", "report"):
        result = executor.execute(scenario(false_count, hardness), request, **limits)
        assert result.status == ("semantic_postcondition_miss" if hardness == "hard" else "qualified"), (
            hardness, result.execution_receipt,
        )
        assert result.proof["completionStatus"] == "exact_complete"
        assert result.proof["claimSatisfied"] is True
        assert result.proof["winningRootActionCount"] == actual_count
        if hardness == "hard":
            assert result.predicate_evaluation.satisfied is False
            assert result.predicate_evaluation.named_core
            continue
        assert result.predicate_evaluation.satisfied is True
        evidence = next(row for row in result.predicate_evaluation.evidence
                        if row["predicateId"] == "PRED-ROOT-COUNT")
        assert evidence["satisfied"] is False

    false_claim = request.to_dict()
    false_claim.pop("sha256")
    false_claim.update(claim="outcome_reachable", successThreshold=2)
    failed = executor.execute(base_scenario, ScenarioProofRequestV1.build(false_claim), **limits)
    assert failed.status == "proof_failed", failed.execution_receipt
    assert failed.proof["completionStatus"] == "proof_failed"
    assert failed.proof["claimSatisfied"] is False


@pytest.mark.parametrize("solver_status", [cp_model.UNKNOWN, cp_model.FEASIBLE, cp_model.INFEASIBLE, cp_model.MODEL_INVALID])
def test_native_complete_state_solver_failures_preserve_outcome_boundary(
    root, monkeypatch, solver_status,
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    final_action = prefix["playActions"].pop()
    scenario = _scenario(
        origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        decision_point={"kind": "play", "selector": {
            "trickIndex": 7, "actingSeat": final_action["actor"],
        }},
        constraints=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-ROOT-COUNT", "proof.winning_root_action_count",
                      {"minimum": 0, "maximum": 1}),
    )
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": "north_south", "utility": "round_win", "successThreshold": 1,
        "budgetProfile": "test_exact",
    })

    class StoppedSolver:
        @staticmethod
        def status_name(status):
            return {
                cp_model.UNKNOWN: "UNKNOWN", cp_model.FEASIBLE: "FEASIBLE",
                cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
            }[status]

    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "_solve",
                        staticmethod(lambda _model, **_kwargs: (StoppedSolver(), solver_status)))
    result = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-NATIVE-FAILURE-TEST", bundle_hash="a" * 64,
    ).execute(scenario, request, maximum_seconds=15, maximum_proof_nodes=1000,
              maximum_proof_depth=1, maximum_hidden_worlds=2)

    model_error = solver_status in {cp_model.INFEASIBLE, cp_model.MODEL_INVALID}
    assert result.status == ("compiler_disagreement" if model_error else "incomplete")
    assert result.proof["completionStatus"] == ("model_error" if model_error else "proof_incomplete")
    assert result.proof["claimSatisfied"] is False
    assert result.proof["winningRootActionCount"] is None
    assert result.proof["winningRootActionLowerBound"] == 0
    assert result.proof["winningRootActionUpperBound"] == 1
    assert result.proof["rootActions"] == []
    assert result.proof.get("nativeQualification") is None
    assert result.predicate_evaluation is None
    assert result.rejection_hashes == ()
    documents = result.semantic_artifact_documents()
    assert canonical_sha256(documents["proofOutcome"]) == result.research_authority["proofOutcomeHash"]


def test_historical_small_card_big_gain_uses_root_value_variables(
    rules,
    complete_candidate,
):
    matching = _matching_scenario(complete_candidate)
    scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                matching.constraints.to_dict(),
                _leaf(
                    "PRED-TACTIC-SMALL-GAIN",
                    "tactic.verified",
                    {
                        "kind": "small_card_big_gain",
                        "minimumScore": 100,
                        "minimumEvidence": 1,
                    },
                ),
            ],
        },
        outcome=matching.outcome.to_dict(),
    )

    proof_value = _bounded_proof(
        claim="selected_action_is_unique_winner",
        selected_action="D7",
    ).to_dict()
    proof_value.pop("sha256")
    proof_value["targetTeam"] = "west_east"
    result = _executor(rules).execute(
        scenario,
        ScenarioProofRequestV1.build(proof_value),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        graph_compiler=lambda _candidate, _request: _two_action_causal_graph(),
        maximum_seconds=15,
    )

    assert result.status == "qualified"
    assert result.research_authority["backendId"] == "ortools_cp_sat_v2"
    evidence = next(
        row
        for row in result.predicate_evaluation.evidence
        if row["predicateId"] == "PRED-TACTIC-SMALL-GAIN"
    )
    assert evidence["observed"]["qualificationSource"] == (
        "native_cp_sat_root_value_variables"
    )
    assert evidence["observed"]["rootValueVariables"] == ["value_0", "value_1"]


def test_campaign_honors_five_results_and_carries_full_candidate_no_goods(
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)
    supplied_exclusions: list[tuple[tuple[int, ...], ...]] = []

    def decision_state(offset: int):
        return {
            "contract": "sun",
            "trumpSuit": None,
            "currentPlayer": "north",
            "buyer": "west",
            "dealer": "east",
            "hands": {
                seat: [
                    Card(suit * 8 + (offset + rank) % 8).to_dict() for rank in range(3)
                ]
                for suit, seat in enumerate(("north", "west", "south", "east"))
            },
            "currentTrick": None,
        }

    def candidate_compiler(_scenario, excluded, **_kwargs):
        supplied_exclusions.append(tuple(excluded))
        offset = len(excluded)
        no_good = tuple((card + offset) % 32 for card in range(32))
        candidate = OrToolsNeutralCandidateV1(
            complete_candidate.origin,
            complete_candidate.trace,
            decision_state(offset),
            no_good,
            complete_candidate.stage_receipts,
            complete_candidate.exact_state_id,
        )
        return OrToolsCandidateAttemptV1("feasible", candidate)

    campaign = _executor(rules).execute_campaign(
        scenario,
        _bounded_proof(claim="selected_action_is_unique_winner", selected_action="D7"),
        result_count=5,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        candidate_compiler=candidate_compiler,
        graph_compiler=lambda _candidate, _request: _two_action_graph(bounded=True),
        maximum_candidates=5,
        maximum_seconds=15,
    )

    assert isinstance(campaign, OrToolsResearchCampaignResultV1)
    assert campaign.status == "completed", json.dumps(campaign.to_dict(), indent=2)
    assert len(campaign.executions) == len(campaign.accepted_results) == 5
    assert len(set(campaign.no_good_vectors)) == 5
    assert [len(item) for item in supplied_exclusions] == [0, 1, 2, 3, 4]
    expected = sorted(
        campaign.executions,
        key=lambda item: _campaign_variant_from_execution_v1(item).semantic_hash,
    )
    assert campaign.accepted_results == tuple(expected)
    assert campaign.to_dict()["completedCandidates"] == 5
    assert campaign.to_dict()["minimumDistance"] == 4
    variant = _campaign_variant_from_execution_v1(campaign.executions[0])
    cuts = _campaign_variant_decision_cuts_v1(variant)
    assert cuts == (tuple(variant.semantic_document["decisionStateCardVectorRaw"]),)


@pytest.mark.parametrize("result_count", [5, 7])
def test_shard_partition_is_independent_from_candidate_attempt_budget(
    monkeypatch,
    rules,
    complete_candidate,
    result_count,
):
    seen: list[tuple[int, int, tuple[int, int, int], int]] = []

    def fake_seed_campaign(self, _scenario, _proof_request, **kwargs):
        seen.append(
            (
                kwargs["seed"],
                kwargs["maximum_candidates"],
                kwargs["candidate_partition"],
                kwargs["result_count"],
            )
        )
        return OrToolsResearchCampaignResultV1(
            "not_found_within_budget",
            (),
            (),
            (),
            (),
            kwargs["result_count"],
            kwargs["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        OrToolsUnifiedResearchExecutorV1,
        "_execute_seed_campaign",
        fake_seed_campaign,
    )
    shards = partition_research_shards(
        run_code="RUN-PARTITION0001",
        run_hash=canonical_sha256("run"),
        recipe_hash=canonical_sha256("recipe"),
        seed_start=100,
        seed_count=10,
        seeds_per_shard=2,
    )
    campaign = _executor(rules).execute_campaign(
        _matching_scenario(complete_candidate),
        _proof(),
        result_count=result_count,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_candidates=10,
        maximum_seconds=15,
        shards=shards,
        shard_worker_count=5,
    )

    assert campaign.status == "not_found_within_budget"
    assert sorted(seen) == [
        (100 + ordinal * 2, 2, (5, ordinal, 1), 1) for ordinal in range(5)
    ]


@pytest.mark.parametrize("seed_count", [1, 4])
@pytest.mark.parametrize("exhausted", (False, True))
def test_v3_builtin_shards_dispatch_outside_the_python_gil(
    monkeypatch,
    rules,
    complete_candidate,
    seed_count,
    exhausted,
):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    seen: list[dict] = []

    def fake_process(task, _request, **_kwargs):
        seen.append(dict(task))
        options = dict(task["options"])
        _request.report_worker_cpu_sample(1_250_000_000)
        if exhausted:
            _request.report_checkpoint({"kind": "proof", "privateHand": [0, 1]})
            _request.report_progress({"currentStage": "shard_budget_exhausted",
                "candidateStatus": "incomplete", "completedCandidates": 1,
                "candidateHash": "b" * 64, "executionOrdinal": 1})
        return OrToolsResearchCampaignResultV1(
            "incomplete" if exhausted else "not_found_within_budget",
            (),
            (),
            (),
            (),
            options["result_count"],
            options["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0 if exhausted else 2.5,
        )

    monkeypatch.setattr(
        executor_module,
        "_run_seed_campaign_process_v1",
        fake_process,
    )
    monkeypatch.setattr(
        executor_module,
        "load_exact_rule_source_catalog",
        lambda *_args: type(
            "Catalog",
            (),
            {"data": {}, "sha256": canonical_sha256("test-rule-sources")},
        )(),
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V3",
        bundle_hash="b" * 64,
        backend_id="ortools_cp_sat_v3",
        witness_policies={"north": {"policyId": "transport-fixture", "seed": 42}},
        tactical_detector_profile=TacticalDetectorProfileV1.defaults(),
    )
    monkeypatch.setattr(
        executor,
        "_execute_seed_campaign",
        lambda *_args, **_kwargs: pytest.fail("V3 fixed shards must run outside the parent"),
    )
    shards = partition_research_shards(
        run_code="RUN-PROCESS0001",
        run_hash=canonical_sha256("run-process"),
        recipe_hash=canonical_sha256("recipe-process"),
        seed_start=10,
        seed_count=seed_count,
        seeds_per_shard=1,
        maximum_attempts=1,
    )

    campaign = executor.execute_campaign(
        _matching_scenario(complete_candidate),
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_candidates=seed_count,
        maximum_seconds=15,
        shards=shards,
        shard_worker_count=2,
    )

    assert campaign.status == ("incomplete" if exhausted else "not_found_within_budget")
    assert len(seen) == seed_count
    expected_cpu_ns = (1_250_000_000 if exhausted else 2_500_000_000) * seed_count
    assert campaign.cpu_seconds >= expected_cpu_ns / 1e9
    assert campaign.coordinator_receipt is not None
    if exhausted:
        assert campaign.executions == ()
        assert all(item.partial_evidence["completedCandidates"] == 1
            for item in campaign.coordinator_receipt.attempts)
        assert all(item.partial_evidence["qualityCheckpoints"]
            for item in campaign.coordinator_receipt.attempts)
    assert campaign.coordinator_receipt.cpu_time_ns == expected_cpu_ns
    assert campaign.coordinator_receipt.cpu_time_coverage == (
        "sampled_lower_bound" if exhausted else "terminal_measurements")
    assert campaign.coordinator_receipt.budget.shard_worker_count == min(seed_count, 2)
    assert all(item["backendId"] == "ortools_cp_sat_v3" for item in seen)
    assert all(item["witnessPolicies"] == executor.witness_policies for item in seen)
    assert all(item['tacticalDetectorProfile'] == executor.tactical_detector_profile for item in seen)
    assert all(item["ruleSourceCatalog"] == {
        "data": executor.rule_source_catalog.data,
        "sha256": executor.rule_source_catalog.sha256,
    } for item in seen)
    assert all(
        0 < item["options"]["maximum_campaign_seconds"] <= 15.0
        for item in seen
    )
    partitions = [item["options"]["candidate_partition"] for item in seen]
    assert partitions.count(None) == 1
    assert sorted(item for item in partitions if item is not None) == [
        (seed_count, ordinal, 1) for ordinal in range(1, seed_count)
    ]


@pytest.mark.parametrize("policies", [None, {"north": {"policyId": "transport-fixture", "seed": 42}}])
@pytest.mark.parametrize('has_detector_profile', [False, True])
def test_shard_child_preserves_witness_policies(monkeypatch, policies, has_detector_profile):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    received = {}
    messages = []
    closed = []

    def executor(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(_execute_seed_campaign=lambda *args, **kwargs: "receipt")

    monkeypatch.setattr(executor_module, "OrToolsUnifiedResearchExecutorV1", executor)
    connection = SimpleNamespace(
        poll=lambda: False, send=messages.append, close=lambda: closed.append(True),
    )
    task = dict(rules="rules", bundleRevisionId="CBR-TEST", bundleHash="a" * 64,
                backendId="ortools_cp_sat_v3", scenario="scenario", proofRequest="proof", options={})
    if policies is not None:
        task["witnessPolicies"] = policies
    if has_detector_profile:
        task['tacticalDetectorProfile'] = TacticalDetectorProfileV1.defaults()
    task["ruleSourceCatalog"] = {"data": {"catalog": "frozen-fixture"}, "sha256": "b" * 64}
    executor_module._execute_seed_campaign_process_v1(connection, task)
    assert received["witness_policies"] == policies
    assert received['tactical_detector_profile'] == task.get('tacticalDetectorProfile')
    assert received["rule_source_catalog"] == task["ruleSourceCatalog"]
    assert messages == [("result", "receipt")]
    assert closed == [True]


def test_v3_process_transport_broken_pipe_is_retried_once(
    monkeypatch,
    rules,
    complete_candidate,
    deterministic_work,
):
    calls: list[tuple[int, int, int]] = []
    budgets: list[float] = []

    def fake_process(task, _request, **_kwargs):
        partition = tuple(task["options"]["candidate_partition"])
        calls.append(partition)
        if partition == (2, 0, 1) and calls.count(partition) == 1:
            budgets.append(task["options"]["maximum_campaign_seconds"])
            deterministic_work.charge(1)
            raise BrokenPipeError("transient process transport")
        if partition == (2, 0, 1):
            budgets.append(task["options"]["maximum_campaign_seconds"])
        options = dict(task["options"])
        return OrToolsResearchCampaignResultV1(
            "not_found_within_budget",
            (),
            (),
            (),
            (),
            options["result_count"],
            options["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        executor_module,
        "_run_seed_campaign_process_v1",
        fake_process,
    )
    monkeypatch.setattr(
        executor_module,
        "load_exact_rule_source_catalog",
        lambda *_args: type(
            "Catalog",
            (),
            {"data": {}, "sha256": canonical_sha256("test-rule-sources")},
        )(),
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V3",
        bundle_hash="b" * 64,
        backend_id="ortools_cp_sat_v3",
    )
    shards = partition_research_shards(
        run_code="RUN-PIPERETRY01",
        run_hash=canonical_sha256("pipe-retry-run"),
        recipe_hash=canonical_sha256("pipe-retry-recipe"),
        seed_start=10,
        seed_count=2,
        seeds_per_shard=1,
        maximum_attempts=1,
    )

    campaign = executor.execute_campaign(
        _matching_scenario(complete_candidate),
        _proof(),
        result_count=2,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_candidates=2,
        maximum_seconds=15,
        shards=shards,
        shard_worker_count=2,
    )

    assert campaign.status == "not_found_within_budget"
    assert sorted(calls) == [(2, 0, 1), (2, 0, 1), (2, 1, 1)]
    assert len(budgets) == 2 and budgets[1] == pytest.approx(budgets[0] - 1)
    assert campaign.coordinator_receipt is not None
    assert {item.status for item in campaign.coordinator_receipt.attempts} == {
        "completed"
    }


@pytest.mark.real_clock
@pytest.mark.parametrize("terminal", [True, False])
@pytest.mark.parametrize("cancel_before_receive", (False, True))
def test_process_parent_retains_private_checkpoint_without_forwarding_it(
    monkeypatch, terminal, cancel_before_receive,
):
    private = {"stage": "project", "artifact": {"privateHand": [0, 1]}}
    messages = [("cpu_sample", 42), ("progress", {"currentStage": "playingQuality"}), ("checkpoint", private)]
    if terminal:
        result = SimpleNamespace(executions=())
        messages.append(("result", result))
    class Pipe:
        def poll(self, *_args):
            return bool(messages)
        def recv(self):
            return messages.pop(0)
        def close(self):
            pass
    class Context:
        def Pipe(self, **_kwargs):
            return Pipe(), Pipe()
        def Process(self, **_kwargs):
            return SimpleNamespace(start=lambda: None, is_alive=lambda: False,
                join=lambda **_kwargs: None)
    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _name: Context())
    live, checkpoints, cpu_samples = [], [], []
    cancelled = False
    class TransferLock:
        def __enter__(self):
            nonlocal cancelled
            cancelled = cancel_before_receive
        def __exit__(self, *_args):
            pass
    request = SimpleNamespace(report_progress=live.append, report_checkpoint=checkpoints.append,
        report_worker_cpu_sample=cpu_samples.append,
        evidence_transfer_lock=TransferLock(),
        cancellation_requested=lambda: cancelled)
    if cancel_before_receive:
        with pytest.raises(RuntimeError, match="process cancelled"):
            executor_module._run_seed_campaign_process_v1({}, request,
                cancellation_poll_milliseconds=5)
        assert live == checkpoints == cpu_samples == []
        assert len(messages) == (4 if terminal else 3)
        return
    if terminal:
        assert executor_module._run_seed_campaign_process_v1({}, request,
            cancellation_poll_milliseconds=5) is result
    else:
        with pytest.raises(RuntimeError, match="without a terminal result"):
            executor_module._run_seed_campaign_process_v1({}, request,
                cancellation_poll_milliseconds=5)
    assert live == [{"currentStage": "playingQuality"}]
    assert checkpoints == [private]
    assert cpu_samples == [42]


@pytest.mark.real_clock
@pytest.mark.parametrize("candidate", (False, True))
def test_v3_process_parent_enforces_shard_budget(monkeypatch, candidate):
    clock = iter((0.0, 0.6, 1.2, 1.3, 1.5, 1.6, 1.7))
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: next(clock))
    class Pipe:
        def poll(self, *_args):
            return False
        def send(self, _value):
            pass
        def close(self):
            pass
    class Process:
        alive = True
        def start(self):
            pass
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.alive = False
        def join(self, **_kwargs):
            pass
    process = Process()
    class Context:
        def Pipe(self, **_kwargs):
            return Pipe(), Pipe()
        def Process(self, **_kwargs):
            return process
    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _name: Context())
    progress = []
    snapshot = ({"currentStage": "playingQuality", "completedCandidates": 1,
        "candidateHash": "b" * 64, "qualitySolver": {"proofPhase": "fallback", "queryCount": 49},
        "qualityCheckpoints": {"private": {"hands": [0, 1]}}, "hands": [0, 1]}
        if candidate else {})
    request = SimpleNamespace(report_progress=progress.append,
        report_checkpoint=lambda _payload: None, cancellation_requested=lambda: False,
        evidence_transfer_lock=executor_module.threading.RLock(),
        progress_snapshot=lambda: snapshot)
    task = {"backendId": "ortools_cp_sat_v3", "options": {
        "maximum_campaign_seconds": 1.0, "result_count": 1,
        "maximum_candidates": 1, "symmetry_policy_id": "rules_preserving_v1",
        "minimum_principal_variation_action_distance": None,
        "minimum_decision_state_distance": 4,
    }}

    result = executor_module._run_seed_campaign_process_v1(
        task, request, cancellation_poll_milliseconds=5)

    assert result.status == "incomplete"
    assert result.executions == ()
    assert not process.alive
    if candidate:
        assert progress == [{"currentStage": "shard_budget_exhausted",
            "candidateHash": "b" * 64, "completedCandidates": 1,
            "qualitySolver": {"proofPhase": "fallback", "queryCount": 49},
            "candidateStatus": "incomplete", "executionOrdinal": 1,
            "candidateAttemptReason": "shard_budget_exhausted",
            "incompleteCompilation": {"stage": "playingQuality", "exhaustionReason": "maximum_seconds"}}]
        from baloot_platform.research.service import _operator_candidate_progress_receipt_v1
        receipt = _operator_candidate_progress_receipt_v1(progress[0])
        assert receipt["classification"] == "incomplete"
        assert receipt["candidateHash"] == "b" * 64
        assert snapshot["qualityCheckpoints"]["private"]["hands"] == [0, 1]
    else:
        assert progress == [{"currentStage": "shard_budget_exhausted"}]


@pytest.mark.parametrize("poll_breaks", (False, True))
def test_v3_process_does_not_replay_a_receipted_incomplete_candidate(monkeypatch, poll_breaks):
    progress = []
    private = {"qualityCheckpoints": {"proof": {"privateHand": [0, 1]}},
               "workerCpuAccounting": {"cpuTimeNs": 42}}
    messages = [("progress", {
        "currentStage": "candidate_terminal",
        "candidateStatus": "incomplete",
        "candidateAttemptReason": "tail_boundary_budget",
        "executionOrdinal": 1,
    })]

    class Pipe:
        def poll(self, *_args):
            if poll_breaks and not messages:
                raise BrokenPipeError("closed pipe")
            return bool(messages)
        def recv(self):
            return messages.pop(0)
        def close(self):
            pass

    class Context:
        def Pipe(self, **_kwargs):
            return Pipe(), Pipe()
        def Process(self, **_kwargs):
            return SimpleNamespace(start=lambda: None, is_alive=lambda: False,
                                   join=lambda **_kwargs: None, exitcode=1)

    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _name: Context())
    request = SimpleNamespace(
        report_progress=progress.append,
        report_checkpoint=lambda _payload: None,
        evidence_transfer_lock=executor_module.threading.RLock(),
        progress_snapshot=lambda: {**progress[-1], **private},
        cancellation_requested=lambda: False,
    )
    task = {"backendId": "ortools_cp_sat_v3", "options": {
        "maximum_campaign_seconds": 300.0,
        "result_count": 1,
        "maximum_candidates": 1,
        "symmetry_policy_id": "rules_preserving_v1",
        "minimum_principal_variation_action_distance": None,
        "minimum_decision_state_distance": 4,
    }}

    result = executor_module._run_seed_campaign_process_v1(
        task, request, cancellation_poll_milliseconds=5,
    )

    assert result.status == "incomplete"
    assert result.executions == ()
    assert progress[-1]["currentStage"] == "transport_after_incomplete"
    assert progress[-1]["candidateAttemptReason"] == "tail_boundary_budget"
    assert all('qualityCheckpoints' not in row and 'workerCpuAccounting' not in row for row in progress)
    assert private['qualityCheckpoints']['proof']['privateHand'] == [0, 1]


@pytest.mark.real_clock
@pytest.mark.parametrize("failed", (False, True))
def test_process_worker_relays_progress_and_honors_cancel(monkeypatch, failed):
    sent: list[tuple[str, object]] = []

    class Connection:
        def __init__(self):
            self.cancel = 2
            self.polls = 0

        def poll(self):
            self.polls += 1
            assert self.polls < 10, "cancellation must drain repeated messages"
            return self.cancel

        def recv(self):
            self.cancel -= 1
            return "cancel"

        def send(self, value):
            sent.append(value)

        def close(self):
            return None

    class Executor:
        def __init__(self, **_kwargs):
            pass

        def _execute_seed_campaign(
            self, _scenario, _proof_request, *, cancel_check, progress_callback,
            checkpoint_callback, **_kwargs
        ):
            assert cancel_check() is True
            progress_callback({"currentStage": "cancelled"})
            checkpoint_callback({"kind": "proof", "stage": "play", "artifact": "old"})
            assert [row for row in sent if row[0] != "cpu_sample"] == [("checkpoint", executor_module.canonical_json({"kind": "proof", "stage": "play", "artifact": "old"}))]
            checkpoint_callback({"kind": "proof", "stage": "play", "artifact": "private"})
            if failed:
                raise ValueError("readback failed")
            return "cancelled-result"

    monkeypatch.setattr(executor_module, "OrToolsUnifiedResearchExecutorV1", Executor)
    clock = iter((100, 200, 300, 400))
    monkeypatch.setattr(executor_module.time, "process_time_ns", lambda: next(clock))
    executor_module._execute_seed_campaign_process_v1(
        Connection(),
        {
            "rules": object(),
            "bundleRevisionId": "CBR-TEST",
            "bundleHash": "a" * 64,
            "backendId": "ortools_cp_sat_v3",
            "scenario": object(),
            "proofRequest": object(),
            "options": {},
        },
    )

    terminal = ("error", {"type": "builtins.ValueError", "detail": "readback failed"}) if failed else ("result", "cancelled-result")
    assert [value for kind, value in sent if kind == "cpu_sample"] == [100, 200]
    assert [row for row in sent if row[0] != "cpu_sample"] == [("checkpoint", executor_module.canonical_json({"kind": "proof", "stage": "play", "artifact": "old"})),
                    ("checkpoint", executor_module.canonical_json({"kind": "proof", "stage": "play", "artifact": "private"})), terminal]


@pytest.mark.parametrize("receive_error,ignore_terminate", ((EOFError, False), (OSError, True)))
def test_process_cancellation_bounds_a_stalled_private_frame(
    monkeypatch, deterministic_work, receive_error, ignore_terminate,
):
    from baloot_platform.research.campaign_coordinator import (
        CampaignCancellationTokenV1, CampaignShardRequestV1,
    )
    released = executor_module.threading.Event()
    kills = []
    cancellation = CampaignCancellationTokenV1()
    shards = partition_research_shards(run_code="RUN-STALLED-PIPE-CANCEL",
        run_hash=canonical_sha256("stalled-pipe"), recipe_hash=canonical_sha256("stalled-recipe"),
        seed_start=0, seed_count=1, seeds_per_shard=1, maximum_attempts=1)
    request = CampaignShardRequestV1(shards[0], 1, cancellation,
        deadline_ns=executor_module.time.perf_counter_ns() + 100_000_000)

    class Pipe:
        def poll(self, *_args):
            return True
        def recv(self):
            cancellation._cancel()
            deterministic_work.charge(0.125)
            released.wait()
            raise receive_error("truncated private frame")
        def close(self):
            pass
    class Process:
        def start(self):
            pass
        def is_alive(self):
            return not released.is_set()
        def terminate(self):
            if not ignore_terminate:
                released.set()
        def kill(self):
            kills.append(True)
            released.set()
        def join(self, **_kwargs):
            pass
    class Context:
        def Pipe(self, **_kwargs):
            return Pipe(), Pipe()
        def Process(self, **_kwargs):
            return Process()
    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _name: Context())

    with pytest.raises(RuntimeError, match="process cancelled"):
        executor_module._run_seed_campaign_process_v1({}, request,
            cancellation_poll_milliseconds=5)
    assert released.is_set() and request.cancellation_acknowledgement.is_set()
    assert kills == ([True] if ignore_terminate else [])
    evidence = request.progress_snapshot()
    assert evidence["incompleteCompilation"] == {
        "stage": "private_evidence_transfer", "exhaustionReason": "maximum_seconds",
    }
    assert "qualityCheckpoints" not in evidence


def test_process_cancellation_snapshot_does_not_wait_for_live_callback(monkeypatch):
    from baloot_platform.research.campaign_coordinator import (
        CampaignCancellationTokenV1, CampaignShardRequestV1,
    )
    entered, released, finished = (executor_module.threading.Event() for _ in range(3))
    errors = []
    def progress(_payload):
        entered.set()
        try:
            released.wait()
        finally:
            finished.set()
    cancellation = CampaignCancellationTokenV1()
    shard = partition_research_shards(run_code="RUN-CALLBACK-CANCEL",
        run_hash=canonical_sha256("callback-pipe"), recipe_hash=canonical_sha256("callback-recipe"),
        seed_start=0, seed_count=1, seeds_per_shard=1, maximum_attempts=1)[0]
    request = CampaignShardRequestV1(shard, 1, cancellation, progress_callback=progress,
        deadline_ns=executor_module.time.perf_counter_ns() + 5_000_000_000)
    messages = [("progress", {"currentStage": "playingQuality"})]
    pipe = SimpleNamespace(poll=lambda *_: bool(messages), recv=lambda: messages.pop(0),
        send=lambda _: None, close=lambda: None)
    process = SimpleNamespace(start=lambda: None, is_alive=lambda: False,
        join=lambda **_: None)
    context = SimpleNamespace(Pipe=lambda **_: (pipe, pipe), Process=lambda **_: process)
    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _: context)
    def run():
        try:
            executor_module._run_seed_campaign_process_v1({}, request,
                cancellation_poll_milliseconds=5)
        except RuntimeError as error:
            errors.append(str(error))
    worker = executor_module.threading.Thread(target=run)
    worker.start()
    try:
        entered.wait()
        cancellation._cancel()
        request.cancellation_acknowledgement.wait()
        # Check the callback does not own the lock before taking a snapshot.
        acquired = request.evidence_transfer_lock.acquire(blocking=False)
        if acquired:
            request.evidence_transfer_lock.release()
        assert acquired
        assert request.progress_snapshot()["currentStage"] == "playingQuality"
        assert not finished.is_set()
    finally:
        released.set()
        worker.join()
    assert errors == ["campaign shard process cancelled"]


def _large_private_checkpoint_transport_worker(connection, _task):
    """Real spawned writer; repeated nested proof rows model a large receipt."""
    row = {"observationHash": "a" * 64, "actions": ["C7", "HA"], "status": "incomplete"}
    encoded_row = executor_module.canonical_json(row)
    payload = '{"artifact":{"queries":[' + ','.join([encoded_row] * 350_000) + ']}}'
    assert 40_000_000 <= len(payload) <= 50_000_000
    try:
        connection.send(("ready", None))
        connection.send(("checkpoint", payload))
        connection.recv()
    finally:
        connection.close()


@pytest.mark.parametrize("snapshot_during_receive,poll_milliseconds", ((False, 5), (True, 250)))
def test_process_cancellation_during_large_private_checkpoint_transfer(
    monkeypatch, snapshot_during_receive, poll_milliseconds,
):
    import multiprocessing
    import threading
    from baloot_platform.research.campaign_coordinator import (
        CampaignBudgetV1, CampaignShardRequestV1, DeterministicCpSatCampaignCoordinatorV1,
    )
    context = multiprocessing.get_context("spawn")
    cancellation = threading.Event()
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_large_private_checkpoint_transport_worker,
        args=(child, {}), daemon=False)
    live = []
    snapshot_started = threading.Event()
    worker_finished = threading.Event()
    original_snapshot = CampaignShardRequestV1.progress_snapshot

    def snapshot(request):
        snapshot_started.set()
        return original_snapshot(request)

    monkeypatch.setattr(CampaignShardRequestV1, "progress_snapshot", snapshot)
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(
        **{**vars(executor_module.time), "monotonic": time.monotonic}))

    class ReceivingPipe:
        def __init__(self, pipe):
            self.pipe = pipe
        def __getattr__(self, name):
            return getattr(self.pipe, name)
        def recv(self):
            cancellation.set()  # Trigger while the production parent receives the real payload.
            # Let the coordinator observe cancellation before unpickling can hold the GIL.
            self.acknowledged.wait()
            if snapshot_during_receive:
                snapshot_started.wait()
            return self.pipe.recv()

    receiving_pipe = ReceivingPipe(parent)

    class Context:
        def Pipe(self, **kwargs):
            return receiving_pipe, child
        def Process(self, **kwargs):
            return process

    def solve(request):
        receiving_pipe.acknowledged = request.cancellation_acknowledgement
        try:
            return executor_module._run_seed_campaign_process_v1(
                {}, request, cancellation_poll_milliseconds=poll_milliseconds)
        finally:
            worker_finished.set()

    monkeypatch.setattr(executor_module.multiprocessing, "get_context", lambda _name: Context())
    shards = partition_research_shards(run_code="RUN-LARGE-PIPE-CANCEL",
        run_hash=canonical_sha256("large-pipe"), recipe_hash=canonical_sha256("large-recipe"),
        seed_start=0, seed_count=1, seeds_per_shard=1, maximum_attempts=1)
    try:
        # Spawn/import and payload construction are setup, outside the cancellation budget.
        process.start()
        monkeypatch.setattr(process, "start", lambda: None)
        child.close()
        assert parent.recv() == ("ready", None)
        receipt = DeterministicCpSatCampaignCoordinatorV1().run(
            shards=shards, budget=CampaignBudgetV1(1, 1, 15, poll_milliseconds),
            solver=solve,
            cancellation_event=cancellation, progress_callback=live.append)
        assert receipt.status == "cancelled"
        evidence = receipt.attempts[0].partial_evidence
        assert evidence["cancellation"]["acknowledged"] is True
        assert evidence["cancellation"]["latencyNs"] == 0
        checkpoints = evidence["qualityCheckpoints"]
        assert len(checkpoints) == 1
        document = next(iter(checkpoints.values()))
        assert len(document["artifact"]["queries"]) == 350_000
        assert list(checkpoints) == [canonical_sha256(document)]
        assert all("qualityCheckpoints" not in row and "artifact" not in row for row in live)
        worker_finished.wait()
        assert not process.is_alive(), "checkpoint cancellation leaked an owned worker"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        child.close()


def test_v3_process_campaign_honors_caller_cancel_after_child_progress(root, monkeypatch):
    import multiprocessing
    import threading

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    scenario = _scenario(
        constraints=_leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
        outcome=_leaf("PRED-OUTCOME", "outcome.team_round",
                      {"team": "north_south", "status": "won"}, "report"),
    )
    shards = partition_research_shards(
        run_code="RUN-PROCESS-CANCEL01", run_hash=canonical_sha256("process-cancel"),
        recipe_hash=scenario.sha256, seed_start=0, seed_count=1,
        seeds_per_shard=1, maximum_attempts=1,
    )
    cancellation = threading.Event()
    worker_finished = threading.Event()
    original_process = executor_module._run_seed_campaign_process_v1

    def track_process(*args, **kwargs):
        try:
            return original_process(*args, **kwargs)
        finally:
            worker_finished.set()

    monkeypatch.setattr(executor_module, "_run_seed_campaign_process_v1", track_process)
    monkeypatch.setattr(executor_module, "time", SimpleNamespace(
        **{**vars(executor_module.time), "monotonic": time.monotonic}))

    def progress(value):
        if value.get("currentStage") == "preflight" and not cancellation.is_set():
            cancellation.set()

    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-TEST-CANCEL", bundle_hash="a" * 64,
    )
    existing_children = {child.pid for child in multiprocessing.active_children()}
    campaign = executor.execute_campaign(
        scenario, _bounded_proof(), result_count=1,
        symmetry_policy_id="rules_preserving_v1", allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4, maximum_candidates=1,
        maximum_seconds=30, maximum_campaign_seconds=30, shards=shards,
        shard_worker_count=1, cancellation_poll_milliseconds=25,
        cancel_check=cancellation.is_set, progress_callback=progress,
    )
    assert cancellation.is_set(), "test must cancel after actual child progress"
    assert campaign.status == "incomplete"
    receipt = campaign.coordinator_receipt
    assert receipt is not None and receipt.status == "cancelled"
    assert len(receipt.attempts) == 1
    attempt = receipt.attempts[0]
    assert attempt.status == "cancelled"
    sample = attempt.partial_evidence["workerCpuAccounting"]
    assert sample["coverage"] == "sampled_lower_bound" and sample["cpuTimeNs"] >= 0
    assert receipt.cpu_time_ns >= sample["cpuTimeNs"]
    assert receipt.to_dict()["cpuTimeCoverage"] == "sampled_lower_bound"
    assert attempt.partial_evidence["cancellation"]["acknowledged"] is True
    assert attempt.partial_evidence["cancellation"]["latencyNs"] == 0
    worker_finished.wait()
    assert not any(
        child.pid not in existing_children for child in multiprocessing.active_children()
    ), "cancelled shard process stayed alive after its parent returned"


def test_sharded_campaign_deep_retry_can_reach_qualified_target(
    monkeypatch,
    rules,
    complete_candidate,
):
    partitions = []
    state = {
        "contract": "sun",
        "trumpSuit": None,
        "currentPlayer": "north",
        "buyer": "west",
        "dealer": "east",
        "hands": {
            seat: [Card(suit * 8 + rank).to_dict() for rank in range(3)]
            for suit, seat in enumerate(("north", "west", "south", "east"))
        },
        "currentTrick": None,
    }
    candidate = OrToolsNeutralCandidateV1(
        complete_candidate.origin,
        complete_candidate.trace,
        state,
        complete_candidate.no_good_vector,
        complete_candidate.stage_receipts,
        complete_candidate.exact_state_id,
    )
    executor = _executor(rules)
    qualified = executor.execute(
        _matching_scenario(candidate),
        _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", candidate
        ),
        graph_compiler=lambda _candidate, _request: _two_action_graph(),
        maximum_proof_nodes=4,
        maximum_proof_depth=4,
        maximum_seconds=15,
    )
    assert qualified.status == "qualified"
    incomplete = replace(
        qualified,
        status="incomplete",
        proof={**dict(qualified.proof or {}), "completionStatus": "proof_incomplete"},
        research_authority=None,
        artifact_preimages=None,
        semantic_hash=canonical_sha256("deep-retry-incomplete"),
    )

    def fake_seed_campaign(self, _scenario, _proof_request, **kwargs):
        partitions.append(kwargs["candidate_partition"])
        if kwargs["maximum_proof_nodes"] < 4:
            return OrToolsResearchCampaignResultV1(
                "incomplete",
                (incomplete,),
                (),
                (),
                (),
                kwargs["result_count"],
                kwargs["maximum_candidates"],
                "rules_preserving_v1",
                "decision_state_card_hamming_v1",
                4,
                0.0,
                0.0,
            )
        return OrToolsResearchCampaignResultV1(
            "completed",
            (qualified,),
            (qualified,),
            (),
            (candidate.no_good_vector,),
            kwargs["result_count"],
            kwargs["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        OrToolsUnifiedResearchExecutorV1,
        "_execute_seed_campaign",
        fake_seed_campaign,
    )
    original_run = executor_module.DeterministicCpSatCampaignCoordinatorV1.run

    def finish_on_budget_boundary(coordinator, **kwargs):
        receipt = original_run(coordinator, **kwargs)
        assert len(receipt.accepted_variants) == 1
        receipt = replace(receipt, status="budget_exhausted", semantic_sha256=None)
        return replace(
            receipt,
            receipt_sha256=canonical_sha256(receipt._receipt_core()),
        )

    monkeypatch.setattr(
        executor_module.DeterministicCpSatCampaignCoordinatorV1,
        "run",
        finish_on_budget_boundary,
    )
    shards = partition_research_shards(
        run_code="RUN-DEEPRETRY001",
        run_hash=canonical_sha256("deep-retry-run"),
        recipe_hash=canonical_sha256("deep-retry-recipe"),
        seed_start=0,
        seed_count=2,
        seeds_per_shard=1,
        maximum_attempts=2,
    )
    campaign = executor.execute_campaign(
        _matching_scenario(candidate),
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_proof_nodes=4,
        maximum_proof_depth=4,
        maximum_candidates=2,
        maximum_seconds=15,
        maximum_campaign_seconds=15,
        shards=shards,
        shard_worker_count=2,
    )

    assert campaign.status == "completed", json.dumps(campaign.to_dict(), indent=2)
    assert campaign.executions[0].status == "incomplete"
    assert campaign.executions[-1].status == "qualified"
    assert len(campaign.accepted_results) == 1
    assert campaign.coordinator_receipt is not None
    assert campaign.coordinator_receipt.budget.shard_worker_count == 2
    assert any(
        item.attempt == 2 for item in campaign.coordinator_receipt.attempts
    )
    assert partitions[0] is None


def test_full_proof_exhaustion_moves_to_next_candidate_within_global_budget(
    monkeypatch,
    rules,
    complete_candidate,
):
    executor = _executor(rules)
    candidate = OrToolsNeutralCandidateV1(
        complete_candidate.origin,
        complete_candidate.trace,
        {
            "contract": "sun",
            "trumpSuit": None,
            "currentPlayer": "north",
            "buyer": "west",
            "dealer": "east",
            "hands": {
                seat: [Card(suit * 8 + rank).to_dict() for rank in range(3)]
                for suit, seat in enumerate(("north", "west", "south", "east"))
            },
            "currentTrick": None,
        },
        complete_candidate.no_good_vector,
        complete_candidate.stage_receipts,
        complete_candidate.exact_state_id,
    )
    qualified = executor.execute(
        _matching_scenario(candidate),
        _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", candidate
        ),
        graph_compiler=lambda _candidate, _request: _two_action_graph(),
        maximum_proof_nodes=4,
        maximum_proof_depth=4,
        maximum_seconds=15,
    )
    incomplete = replace(
        qualified,
        status="incomplete",
        proof={**dict(qualified.proof or {}), "completionStatus": "proof_incomplete"},
        research_authority=None,
        artifact_preimages=None,
        semantic_hash=canonical_sha256("full-proof-exhausted"),
    )
    calls: list[tuple[int, tuple[tuple[int, ...], ...], int]] = []

    def fake_seed_campaign(self, _scenario, _proof_request, **kwargs):
        excluded = tuple(kwargs["initial_excluded_decks"])
        calls.append((kwargs["seed"], excluded, kwargs["maximum_proof_nodes"]))
        if not excluded:
            return OrToolsResearchCampaignResultV1(
                "incomplete",
                (incomplete,),
                (),
                (),
                (),
                kwargs["result_count"],
                kwargs["maximum_candidates"],
                "rules_preserving_v1",
                "decision_state_card_hamming_v1",
                4,
                0.0,
                0.0,
            )
        return OrToolsResearchCampaignResultV1(
            "completed",
            (qualified,),
            (qualified,),
            (),
            (candidate.no_good_vector,),
            kwargs["result_count"],
            kwargs["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        OrToolsUnifiedResearchExecutorV1,
        "_execute_seed_campaign",
        fake_seed_campaign,
    )
    shards = partition_research_shards(
        run_code="RUN-PROOFNEXT001",
        run_hash=canonical_sha256("proof-next-run"),
        recipe_hash=canonical_sha256("proof-next-recipe"),
        seed_start=0,
        seed_count=2,
        seeds_per_shard=2,
        maximum_attempts=3,
    )
    campaign = executor.execute_campaign(
        _matching_scenario(candidate),
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_proof_nodes=4,
        maximum_proof_depth=4,
        maximum_candidates=2,
        maximum_seconds=15,
        maximum_campaign_seconds=15,
        shards=shards,
        shard_worker_count=1,
    )

    assert campaign.status == "completed", json.dumps(campaign.to_dict(), indent=2)
    assert calls == [
        (0, (), 1),
        (0, (), 4),
        (1, (candidate.no_good_vector,), 4),
    ]


def test_native_decision_cut_exhaustion_is_not_global_infeasibility(root, monkeypatch):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    hands = {seat: tuple(Card(8 * int(seat) + rank) for rank in range(8)) for seat in Seat}
    tail_inputs = dict(
        cards=tuple(card for hand in hands.values() for card in hand),
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract="sun", trump=None, first_leader="north",
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        decision_prefix_length=0, decision_actor="north", maximum_time_seconds=5,
        selected_play_count=4,
    )
    stage = OrToolsPlayTailStageV1(native_conversion=True)
    assert stage.solve(**tail_inputs) is not None
    cuts = tuple((status,) * 32 for status in range(9))

    def candidate_compiler(_scenario, excluded, **kwargs):
        assert not excluded
        with pytest.raises(PlayTailSolveStopped, match="infeasible"):
            stage.solve(**tail_inputs, excluded_decision_status_vectors=cuts,
                        minimum_decision_status_hamming_distance=32)
        return OrToolsCandidateAttemptV1("infeasible", None, "tail_first_infeasible")

    def tail_factory(**kwargs):
        assert kwargs["excluded_decision_status_vectors"] == cuts
        assert kwargs["minimum_decision_status_hamming_distance"] == 32
        return candidate_compiler

    monkeypatch.setattr(
        "baloot_platform.research.ortools_tail_first_candidate.OrToolsTailFirstCandidateCompilerV1",
        tail_factory,
    )
    scenario = _scenario(
        constraints={"op": "all", "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-CHOICES", "state.legal_action_count", {"minimum": 2, "maximum": 2})]},
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2, "actingSeat": "north"}},
        outcome=_leaf("PRED-REPORT", "outcome.team_round",
                      {"team": "north_south", "status": "won"}, "report"),
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-CUT-EXHAUSTION", bundle_hash="a" * 64,
    )
    campaign = executor._execute_seed_campaign(
        scenario, _proof(), result_count=1, symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False, minimum_decision_state_distance=32,
        maximum_candidates=1, maximum_seconds=15, maximum_proof_nodes=100,
        maximum_proof_depth=8, maximum_prefix_lift_attempts=1,
        initial_excluded_decision_status_vectors=cuts,
    )
    assert campaign.executions[0].status == "not_found"
    assert campaign.status == "not_found_within_budget"
    assert campaign.executions[0].execution_receipt["candidateAttemptReason"] == "tail_first_infeasible"


@pytest.mark.parametrize("backend", ["ortools_cp_sat_v2", "ortools_cp_sat_v3"])
@pytest.mark.parametrize("cut_kind", ["none", "deck", "decision", "quality"])
def test_construction_infeasibility_scope_preserves_uncut_and_legacy(
    root, monkeypatch, complete_candidate, backend, cut_kind,
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _scenario(
        constraints={"op": "all", "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-CHOICES", "state.legal_action_count", {"minimum": 2, "maximum": 2})]},
        decision_point={"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2, "actingSeat": "north"}},
        outcome=_leaf("PRED-REPORT", "outcome.team_round",
                      {"team": "north_south", "status": "won"}, "report"),
    )
    quality_cuts = []

    def tail_factory(**_kwargs):
        def compile_candidate(*_args, **_options):
            return OrToolsCandidateAttemptV1("infeasible", None, "construction_infeasible")

        compile_candidate._reject_quality_candidate = quality_cuts.append
        return compile_candidate

    monkeypatch.setattr(
        "baloot_platform.research.ortools_tail_first_candidate.OrToolsTailFirstCandidateCompilerV1",
        tail_factory,
    )
    result = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id=backend, bundle_revision_id="CBR-INFEASIBILITY-SCOPE",
        bundle_hash="a" * 64,
    ).execute(
        scenario, _proof(), maximum_candidates=1, maximum_seconds=5,
        maximum_proof_nodes=100, maximum_proof_depth=8, maximum_prefix_lift_attempts=1,
        initial_excluded_decks=(tuple(range(32)),) if cut_kind == "deck" else (),
        excluded_decision_status_vectors=((0,) * 32,) if cut_kind == "decision" else (),
        initial_quality_rejected_candidates=(complete_candidate,) if cut_kind == "quality" else (),
    )
    expected = "not_found" if backend == "ortools_cp_sat_v3" and cut_kind != "none" else "infeasible"
    assert result.status == expected
    assert quality_cuts == ([complete_candidate] if cut_kind == "quality" else [])


def test_seed_campaign_applies_replenishment_deck_no_goods_before_construction(
    rules,
    complete_candidate,
):
    initial_deck = tuple(range(32))
    initial_vector = tuple(index % 4 for index in range(32))
    seen: list[tuple[tuple[int, ...], ...]] = []

    def candidate_compiler(_scenario, excluded, **kwargs):
        seen.append(tuple(excluded))
        return OrToolsCandidateAttemptV1("not_found", None, "partition_exhausted")

    campaign = _executor(rules)._execute_seed_campaign(
        _matching_scenario(complete_candidate),
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        candidate_compiler=candidate_compiler,
        maximum_candidates=1,
        maximum_seconds=15,
        initial_excluded_decks=(initial_deck,),
        initial_excluded_decision_status_vectors=(initial_vector,),
    )

    assert campaign.status == "not_found_within_budget"
    assert seen == [(initial_deck,)]


@pytest.mark.parametrize("tail_first", [False, True])
@pytest.mark.parametrize("miss", ["quality", "curated", "ordinary_not_found", "curated_empty"])
def test_seed_campaign_carries_only_exact_deck_cut_after_quality_miss(
    rules,
    complete_candidate,
    monkeypatch,
    tail_first,
    miss,
) -> None:
    executor = _executor(rules)
    scenario = _matching_scenario(complete_candidate)
    qualified = executor.execute(
        scenario,
        _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        graph_compiler=lambda *_args: _two_action_graph(),
        maximum_seconds=15,
    )
    assert qualified.status == "qualified"
    assert qualified.candidate is not None
    carried_candidate = qualified.candidate
    if tail_first:
        carried_candidate = replace(carried_candidate, trace={
            **carried_candidate.trace,
            "contract": "sun",
            "tailFirstStagePlan": {"sha256": "a" * 64},
        })
    executor.backend_id = "ortools_cp_sat_v3"
    calls = []

    def execute(*_args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return replace(
                qualified,
                status="semantic_postcondition_miss" if miss == "quality" else "not_found",
                candidate=carried_candidate,
                proof=None,
                research_authority=None,
                artifact_preimages=None,
                execution_receipt={
                    "completedCandidates": 0 if miss == "curated_empty" else 1,
                    "candidateAttemptReason": (
                        "quality_generation_target_miss" if miss == "quality" else
                        "curated_interest_miss" if miss in {"curated", "curated_empty"} else "other_miss"),
                    "incompleteCompilation": {
                        "qualityPolicyCut": {
                            "kind": "play_divergence_v1",
                            "playIndex": 0,
                            "samePolicyObservation": True,
                        }
                    },
                },
                semantic_hash=canonical_sha256("quality-miss"),
            )
        return replace(
            qualified,
            status="infeasible",
            candidate=None,
            proof=None,
            research_authority=None,
            artifact_preimages=None,
            execution_receipt={"completedCandidates": 0},
            semantic_hash=canonical_sha256("campaign-stop"),
        )

    monkeypatch.setattr(executor, "execute", execute)
    executor._execute_seed_campaign(
        scenario,
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=2,
        maximum_candidates=2,
        maximum_seconds=15,
        maximum_campaign_seconds=30,
    )

    if miss in {"ordinary_not_found", "curated_empty"}:
        assert len(calls) == 1
        return
    assert len(calls) == 2
    quality_cut = tail_first and miss == "quality"
    assert calls[1]["initial_excluded_decks"] == ([] if quality_cut else [carried_candidate.no_good_vector])
    assert calls[1]["initial_quality_rejected_candidates"] == (
        (carried_candidate,) if quality_cut else ()
    )


@pytest.mark.parametrize("proposal_queue", [False, True])
def test_shard_retry_carries_tail_quality_play_cut(
    rules, complete_candidate, monkeypatch, proposal_queue,
) -> None:
    executor = _executor(rules)
    scenario = _matching_scenario(complete_candidate)
    qualified = executor.execute(
        scenario, _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate),
        graph_compiler=lambda *_args: _two_action_graph(),
        maximum_seconds=15,
    )
    assert qualified.status == "qualified" and qualified.candidate is not None
    rejected = replace(qualified.candidate, trace={
        **qualified.candidate.trace,
        "contract": "sun",
        **({} if proposal_queue else {"tailFirstStagePlan": {"sha256": "a" * 64}}),
    })
    miss = replace(
        qualified, status="semantic_postcondition_miss", candidate=rejected,
        proof=None, research_authority=None, artifact_preimages=None,
        execution_receipt={"completedCandidates": 1,
                           "candidateAttemptReason": "quality_generation_target_miss"},
        semantic_hash=canonical_sha256("shard-quality-miss"),
    )
    incomplete = replace(
        qualified, status="incomplete", candidate=None, proof=None,
        research_authority=None, artifact_preimages=None,
        execution_receipt={"completedCandidates": 0},
        semantic_hash=canonical_sha256("shard-quality-incomplete"),
    )
    calls = []

    def fake_seed_campaign(self, _scenario, _proof_request, **kwargs):
        calls.append(kwargs)
        first = len(calls) == 1
        return OrToolsResearchCampaignResultV1(
            "incomplete" if first else "not_found_within_budget",
            (miss, incomplete) if first else (), (), (), (),
            kwargs["result_count"], kwargs["maximum_candidates"],
            "rules_preserving_v1", "decision_state_card_hamming_v1", 4,
            0.0, 0.0,
        )

    monkeypatch.setattr(OrToolsUnifiedResearchExecutorV1, "_execute_seed_campaign", fake_seed_campaign)
    executor.backend_id = "ortools_cp_sat_v3"
    if proposal_queue:
        executor.runtime_proposals = {0: {}, 1: {}, 2: {}}
    monkeypatch.setattr(
        "baloot_platform.research.ortools_research_executor._run_seed_campaign_process_v1",
        lambda task, _request, **_kwargs: fake_seed_campaign(
            None, task["scenario"], task["proofRequest"], **task["options"]),
    )
    shards = partition_research_shards(
        run_code="RUN-QUALITYCUT001", run_hash=canonical_sha256("quality-cut-run"),
        recipe_hash=canonical_sha256("quality-cut-recipe"), seed_start=0,
        seed_count=3, seeds_per_shard=3, maximum_attempts=2,
    )
    campaign = executor.execute_campaign(
        scenario, _proof(), result_count=1,
        symmetry_policy_id="rules_preserving_v1", allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4, maximum_candidates=3,
        maximum_seconds=15, maximum_campaign_seconds=15,
        shards=shards, shard_worker_count=1,
    )
    assert len(calls) == 2
    assert calls[1]["initial_excluded_decks"] == ()
    assert calls[1]["initial_quality_rejected_candidates"] == (() if proposal_queue else (rejected,))
    assert [item.status for item in campaign.coordinator_receipt.attempts] == [
        "completed", "completed"]
    assert calls[1]["seed"] == 2


def test_native_campaign_excludes_incomplete_tail_boundary_state(
    rules, complete_candidate, monkeypatch
) -> None:
    executor = _executor(rules)
    scenario = _matching_scenario(complete_candidate)
    qualified = executor.execute(
        scenario, _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        graph_compiler=lambda *_args: _two_action_graph(),
        maximum_seconds=15,
    )
    executor.backend_id = "ortools_cp_sat_v3"
    cut = tuple(index % 9 for index in range(32))
    calls = []

    def execute(*_args, **kwargs):
        calls.append(kwargs)
        return replace(
            qualified, status="incomplete", candidate=None, proof=None,
            research_authority=None, artifact_preimages=None,
            execution_receipt={
                "completedCandidates": 0,
                "incompleteCompilation": {
                    "stage": "proofBoundary",
                    "evidence": {"decisionStateStatusVector": list(cut)},
                },
            },
            semantic_hash=canonical_sha256(["boundary", len(calls)]),
        )

    monkeypatch.setattr(executor, "execute", execute)
    campaign = executor._execute_seed_campaign(
        scenario, _proof(), result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=2, maximum_candidates=3,
        maximum_seconds=15, maximum_campaign_seconds=60,
    )
    assert campaign.status == "incomplete"
    assert len(calls) == 2
    assert calls[0]["excluded_decision_status_vectors"] == ()
    assert calls[1]["excluded_decision_status_vectors"] == (cut,)


@pytest.mark.real_clock
@pytest.mark.parametrize(
    "stop, expected_calls, expected_status",
    [("qualified", 5, "completed"), ("candidate_incomplete", 5, "completed"),
     ("infeasible", 5, "incomplete"),
     ("candidate_limit", 4, "incomplete"), ("deadline", 3, "incomplete"),
     ("cancelled", 2, "incomplete"), ("legacy", 1, "incomplete")],
)
@pytest.mark.parametrize("entry_point", ["execute_campaign", "_execute_seed_campaign"])
def test_native_seed_campaign_continues_incomplete_within_separate_budgets(
    monkeypatch, rules, complete_candidate, stop, expected_calls, expected_status, entry_point,
):
    candidate = replace(complete_candidate, state={
        "contract": "sun", "trumpSuit": None, "currentPlayer": "north",
        "buyer": "west", "dealer": "east", "currentTrick": None,
        "hands": {
            seat: [Card(suit * 8 + rank).to_dict() for rank in range(3)]
            for suit, seat in enumerate(("north", "west", "south", "east"))
        },
    })
    executor = _executor(rules)
    scenario = _matching_scenario(candidate)
    qualified = executor.execute(
        scenario, _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", candidate
        ),
        graph_compiler=lambda *_args: _two_action_graph(), maximum_seconds=15,
    )
    assert qualified.status == "qualified"
    executor.backend_id = "ortools_cp_sat_v3" if stop != "legacy" else "ortools_cp_sat_v2"
    clock = [0.0]
    seen = []

    def execute(*_args, **kwargs):
        seen.append(kwargs)
        clock[0] += kwargs["maximum_stage_seconds"]
        if len(seen) == 5 and stop in {"qualified", "candidate_incomplete"}:
            return qualified
        retained = (
            replace(candidate, no_good_vector=tuple(range(1, 32)) + (0,))
            if stop == "candidate_incomplete" and len(seen) == 1 else None
        )
        return replace(
            qualified, status="infeasible" if len(seen) == 5 else "incomplete",
            candidate=retained, proof=None, research_authority=None, artifact_preimages=None,
            execution_receipt={"completedCandidates": int(retained is not None)},
            semantic_hash=canonical_sha256([stop, len(seen)]),
        )

    monkeypatch.setattr(executor, "execute", execute)
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: clock[0])
    result = getattr(executor, entry_point)(
        scenario, _proof(), result_count=1, symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False, minimum_decision_state_distance=2,
        maximum_candidates=4 if stop == "candidate_limit" else 8,
        maximum_seconds=10,
        maximum_campaign_seconds=None if stop == "legacy" else (25 if stop == "deadline" else 100),
        cancel_check=lambda: stop == "cancelled" and len(seen) >= 2,
    )
    assert result.status == expected_status
    assert len(seen) == expected_calls
    assert [item["seed"] for item in seen] == list(range(expected_calls))
    assert all(0 < item["maximum_stage_seconds"] <= 10 for item in seen)
    assert all(item["maximum_stage_seconds"] <= item["maximum_seconds"] for item in seen)
    assert len(result.accepted_results) == int(stop in {"qualified", "candidate_incomplete"})
    if stop == "candidate_incomplete":
        assert tuple(range(1, 32)) + (0,) in seen[-1]["initial_excluded_decks"]
    if stop == "deadline":
        assert [item["maximum_stage_seconds"] for item in seen] == [10, 10, 5]
        assert [item["maximum_seconds"] for item in seen] == [25, 15, 5]


def test_sharded_campaign_retries_incomplete_search_with_next_seed(
    monkeypatch,
    rules,
    complete_candidate,
):
    candidate = OrToolsNeutralCandidateV1(
        complete_candidate.origin,
        complete_candidate.trace,
        {
            "contract": "sun",
            "trumpSuit": None,
            "currentPlayer": "north",
            "buyer": "west",
            "dealer": "east",
            "hands": {
                seat: [Card(suit * 8 + rank).to_dict() for rank in range(3)]
                for suit, seat in enumerate(("north", "west", "south", "east"))
            },
            "currentTrick": None,
        },
        complete_candidate.no_good_vector,
        complete_candidate.stage_receipts,
        complete_candidate.exact_state_id,
    )
    qualified = _executor(rules).execute(
        _matching_scenario(candidate),
        _proof(),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", candidate
        ),
        graph_compiler=lambda _candidate, _request: _two_action_graph(),
        maximum_proof_nodes=4,
        maximum_proof_depth=4,
        maximum_seconds=15,
    )
    seen_seeds: list[int] = []

    def fake_seed_campaign(self, _scenario, _proof_request, **kwargs):
        seen_seeds.append(kwargs["seed"])
        if kwargs["seed"] == 0:
            return OrToolsResearchCampaignResultV1(
                "incomplete",
                (),
                (),
                (),
                (),
                kwargs["result_count"],
                kwargs["maximum_candidates"],
                "rules_preserving_v1",
                "decision_state_card_hamming_v1",
                4,
                0.0,
                0.0,
            )
        return OrToolsResearchCampaignResultV1(
            "completed",
            (qualified,),
            (qualified,),
            (),
            (candidate.no_good_vector,),
            kwargs["result_count"],
            kwargs["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        OrToolsUnifiedResearchExecutorV1,
        "_execute_seed_campaign",
        fake_seed_campaign,
    )
    shards = partition_research_shards(
        run_code="RUN-SEARCHRETRY01",
        run_hash=canonical_sha256("search-retry-run"),
        recipe_hash=canonical_sha256("search-retry-recipe"),
        seed_start=0,
        seed_count=2,
        seeds_per_shard=2,
        maximum_attempts=2,
    )
    campaign = _executor(rules).execute_campaign(
        _matching_scenario(candidate),
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_proof_nodes=4,
        maximum_proof_depth=4,
        maximum_candidates=2,
        maximum_seconds=15,
        maximum_campaign_seconds=15,
        shards=shards,
        shard_worker_count=1,
    )

    assert campaign.status == "completed", json.dumps(campaign.to_dict(), indent=2)
    assert seen_seeds == [0, 1]


def test_sharded_campaign_does_not_schedule_vacuous_retry_after_budget_is_spent(
    monkeypatch,
    rules,
    complete_candidate,
):
    candidates = tuple(
        replace(
            complete_candidate,
            no_good_vector=tuple(range(offset, 32)) + tuple(range(offset)),
        )
        for offset in range(3)
    )
    executions = tuple(
        replace(
            _executor(rules).execute(
                _matching_scenario(candidate),
                _proof(),
                candidate_compiler=lambda *_args, _candidate=candidate, **_kwargs: (
                    OrToolsCandidateAttemptV1("feasible", _candidate)
                ),
                graph_compiler=lambda _candidate, _request: _two_action_graph(),
                maximum_seconds=15,
            ),
            status="semantic_postcondition_miss",
        )
        for candidate in candidates
    )
    incomplete = replace(
        executions[0],
        status="incomplete",
        candidate=None,
        proof=None,
        research_authority=None,
        artifact_preimages=None,
        execution_receipt={"completedCandidates": 0},
    )
    calls = 0

    def fake_seed_campaign(self, _scenario, _proof_request, **kwargs):
        nonlocal calls
        calls += 1
        return OrToolsResearchCampaignResultV1(
            "incomplete",
            (*executions, incomplete),
            (),
            (),
            tuple(candidate.no_good_vector for candidate in candidates),
            kwargs["result_count"],
            kwargs["maximum_candidates"],
            "rules_preserving_v1",
            "decision_state_card_hamming_v1",
            4,
            0.0,
            0.0,
        )

    monkeypatch.setattr(
        OrToolsUnifiedResearchExecutorV1,
        "_execute_seed_campaign",
        fake_seed_campaign,
    )
    progress: list[dict] = []
    shards = partition_research_shards(
        run_code="RUN-SPENTBUDGET1",
        run_hash=canonical_sha256("spent-budget-run"),
        recipe_hash=canonical_sha256("spent-budget-recipe"),
        seed_start=0,
        seed_count=4,
        seeds_per_shard=4,
        maximum_attempts=2,
    )

    campaign = _executor(rules).execute_campaign(
        _matching_scenario(candidates[0]),
        _proof(),
        result_count=1,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        maximum_candidates=4,
        maximum_seconds=15,
        maximum_campaign_seconds=60,
        shards=shards,
        shard_worker_count=1,
        progress_callback=lambda value: progress.append(dict(value)),
    )

    assert calls == 1
    assert not any(
        item.get("currentStage") == "attempt_retry_scheduled" for item in progress
    )
    assert campaign.coordinator_receipt is not None
    assert campaign.coordinator_receipt.attempts[0].partial_evidence[
        "candidateBudgetExhausted"
    ] is True


def test_rejected_quality_handoff_keeps_construction_and_final_proofs_distinct(
    root,
):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    request = _proof(claim="winning_root_action_count")
    construction = OrToolsScenarioGraphProofV1().prove(
        _two_action_graph(),
        request,
        maximum_seconds=5,
    )
    final_graph = ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=(
            ProofGraphNodeV1.decision(
                "root",
                Team.NORTH_SOUTH.name.lower(),
                (
                    ProofGraphEdgeV1.build(0, "D7", "lose-1"),
                    ProofGraphEdgeV1.build(1, "D8", "lose-2"),
                ),
                actor_seat="north",
            ),
            ProofGraphNodeV1.leaf("lose-1", -1),
            ProofGraphNodeV1.leaf("lose-2", -1),
        ),
    )
    final = OrToolsScenarioGraphProofV1().prove(
        final_graph,
        request,
        maximum_seconds=5,
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-QUALITY-HANDOFF",
        bundle_hash="a" * 64,
    )
    scenario = _scenario(
        constraints=_leaf(
            "PRED-HANDOFF-STATE",
            "state.phase",
            {"state": "playing"},
            "report",
        ),
        outcome=_leaf(
            "PRED-HANDOFF-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )

    result = executor._result(
        "semantic_postcondition_miss",
        executor.compiler.preflight(scenario),
        None,
        None,
        final.to_dict(),
        (),
        0.0,
        0.0,
        1,
        graph=final_graph,
        proof_outcome=final,
        scenario_hash=scenario.sha256,
        quality_handoff_diagnostic={
            "schemaVersion": 1,
            "kind": "quality_handoff_diagnostic_v1",
            "qualificationAuthority": "generated_quality_trace",
            "constructionCandidateHash": "b" * 64,
            "constructionProofHash": construction.sha256,
            "constructionWinningRootActionCount": (
                construction.winning_root_action_count
            ),
            "generatedCandidateHash": "c" * 64,
            "firstConstructedPlayDivergenceIndex": 3,
        },
    )

    diagnostic = result.execution_receipt["qualityHandoffDiagnostic"]
    assert diagnostic["constructionWinningRootActionCount"] == 1
    assert diagnostic["finalWinningRootActionCount"] == 0
    assert diagnostic["firstConstructedPlayDivergenceIndex"] == 3
    assert diagnostic["qualificationAuthority"] == "generated_quality_trace"
    assert result.proof["sha256"] == final.sha256
    assert result.research_authority["proofOutcomeHash"] != construction.sha256


@pytest.mark.parametrize("dynamic_selector", [False, True])
@pytest.mark.parametrize("native_scoring", [False, True])
def test_decision_diversity_infeasible_preserves_deal_candidate_for_post_gate(
    monkeypatch,
    rules,
    complete_candidate,
    dynamic_selector,
    native_scoring,
):
    document = _scenario(
        constraints=_leaf(
            "PRED-DIVERSITY-REPORT-STATE",
            "state.phase",
            {"state": "playing"},
            "report",
        ),
        outcome=_leaf(
            "PRED-DIVERSITY-REPORT-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    ).to_dict()
    document.pop("sha256")
    selector = {"trickIndex": 6, "position": 0}
    if dynamic_selector:
        selector = {
            "actingSeat": complete_candidate.trace["play"]["plays"][24]["actor"]
        }
    document["decisionPoint"]["selector"] = selector
    scenario = ResearchScenarioV1.build(document)
    original_solve = OrToolsPlayTailStageV1.solve
    cut_calls = []
    conversion_modes = []

    def stop_cut_solve(self, **kwargs):
        has_cut = bool(kwargs["excluded_decision_status_vectors"])
        cut_calls.append(has_cut)
        conversion_modes.append(self.native_conversion)
        if has_cut:
            if dynamic_selector and native_scoring:
                original_solve(
                    self,
                    **{**kwargs, "excluded_decision_status_vectors": ()},
                )
            raise PlayTailSolveStopped("infeasible")
        return original_solve(self, **kwargs)

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", stop_cut_solve)
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules,
        excluded_decision_status_vectors=((0,) * 32,),
        minimum_decision_status_hamming_distance=4,
        native_cp_sat_scoring=native_scoring,
    )(scenario, (), seed=0, maximum_seconds=15)

    assert attempt.status == "feasible"
    assert attempt.candidate is not None
    assert attempt.candidate.no_good_vector
    assert cut_calls == ([False, True] if dynamic_selector else [True, False])
    assert conversion_modes == [native_scoring] * 2


def test_decision_diversity_unknown_remains_incomplete(
    monkeypatch,
    rules,
    complete_candidate,
):
    document = _scenario(
        constraints=_leaf(
            "PRED-DIVERSITY-UNKNOWN-STATE",
            "state.phase",
            {"state": "playing"},
            "report",
        ),
        outcome=_leaf(
            "PRED-DIVERSITY-UNKNOWN-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    ).to_dict()
    document.pop("sha256")
    document["decisionPoint"]["selector"]["actingSeat"] = complete_candidate.trace[
        "play"
    ]["plays"][24]["actor"]
    scenario = ResearchScenarioV1.build(document)
    original_solve = OrToolsPlayTailStageV1.solve
    cut_calls = []

    def stop_cut_solve(self, **kwargs):
        has_cut = bool(kwargs["excluded_decision_status_vectors"])
        cut_calls.append(has_cut)
        if has_cut:
            raise PlayTailSolveStopped("unknown")
        return original_solve(self, **kwargs)

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", stop_cut_solve)
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules,
        excluded_decision_status_vectors=((0,) * 32,),
        minimum_decision_status_hamming_distance=4,
    )(scenario, (), seed=0, maximum_seconds=15)

    assert attempt == OrToolsCandidateAttemptV1(
        "incomplete", None, "decision_diversity_budget"
    )
    assert cut_calls == [False, True]


def test_campaign_rejects_symmetry_only_qualified_candidate(
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)
    state = {
        "contract": "sun",
        "trumpSuit": None,
        "currentPlayer": "north",
        "buyer": "west",
        "dealer": "east",
        "hands": {
            seat: [Card(suit * 8 + rank).to_dict() for rank in range(3)]
            for suit, seat in enumerate(("north", "west", "south", "east"))
        },
        "currentTrick": None,
    }

    def candidate_compiler(_scenario, excluded, **_kwargs):
        offset = len(excluded)
        return OrToolsCandidateAttemptV1(
            "feasible",
            OrToolsNeutralCandidateV1(
                complete_candidate.origin,
                complete_candidate.trace,
                state,
                tuple((card + offset) % 32 for card in range(32)),
                complete_candidate.stage_receipts,
                complete_candidate.exact_state_id,
            ),
        )

    campaign = _executor(rules).execute_campaign(
        scenario,
        _bounded_proof(claim="selected_action_is_unique_winner", selected_action="D7"),
        result_count=2,
        symmetry_policy_id="rules_preserving_v1",
        allow_symmetry_only_variants=False,
        minimum_decision_state_distance=4,
        candidate_compiler=candidate_compiler,
        graph_compiler=lambda _candidate, _request: _two_action_graph(bounded=True),
        maximum_candidates=2,
        maximum_seconds=15,
    )

    assert campaign.status == "not_found_within_budget"
    assert len(campaign.accepted_results) == 1
    assert [item.rejection_reason for item in campaign.variant_receipts].count(
        "symmetry"
    ) == 1


@pytest.mark.real_clock
@pytest.mark.parametrize("attempt_status", ["infeasible", "not_found", "incomplete"])
def test_executor_preserves_nonfeasible_candidate_statuses(
    rules,
    complete_candidate,
    attempt_status,
):
    scenario = _matching_scenario(complete_candidate)

    def candidate_compiler(*_args, **_kwargs):
        return OrToolsCandidateAttemptV1(
            attempt_status, None, "TEST-STOP",
            stage_timings={"prefixReachability": 0.125},
        )

    result = _executor(rules).execute(
        scenario,
        _proof(),
        candidate_compiler=candidate_compiler,
        maximum_seconds=15,
    )
    assert result.status == attempt_status
    assert result.candidate is None
    assert result.execution_receipt["stageTimings"]["prefixReachability"] == 0.125


@pytest.mark.real_clock
@pytest.mark.parametrize("origin", ["search", "exact_trace_prefix"])
def test_preflight_exhaustion_never_starts_construction(root, rules, complete_candidate, monkeypatch, origin):
    scenario = _matching_scenario(complete_candidate)
    if origin == "exact_trace_prefix":
        fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
        scenario = _scenario(origin={"kind": origin, "exactTracePrefix": fixture["exactTracePrefix"]},
            constraints=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}),
            outcome=_leaf("PRED-ACTOR", "state.next_actor", {"seat": fixture["expected"]["actingSeat"]}))
    executor = _executor(rules)
    original = executor.compiler.preflight
    clock = executor_module.time.perf_counter
    done = []
    def preflight(*args, **kwargs):
        result = original(*args, **kwargs)
        done.append(True)
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError("expired campaign must not launch a solver")
    monkeypatch.setattr(executor.compiler, "preflight", preflight)
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: clock() + (16 if done else 0))
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", forbidden)
    result = executor.execute(scenario, _proof(), candidate_compiler=forbidden, maximum_seconds=15)
    assert result.status == "incomplete"
    assert result.candidate is None
    receipt = result.execution_receipt["incompleteCompilation"]
    assert receipt["stage"] == "construction"
    assert receipt["exhaustionReason"] == "maximum_seconds"
    assert receipt["evidence"]["completedCandidates"] == 0
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.real_clock
@pytest.mark.parametrize("stage", ["graph", "proof"])
@pytest.mark.parametrize("reason", ["maximum_seconds", "cancelled"])
def test_complete_graph_handoff_stops_campaign(rules, complete_candidate, monkeypatch, stage, reason):
    ready = []
    graph_calls = []
    clock = executor_module.time.perf_counter
    def candidate(*args, **kwargs):
        if stage == "graph":
            ready.append(True)
        return OrToolsCandidateAttemptV1("feasible", complete_candidate)
    def graph(*args, **kwargs):
        graph_calls.append(True)
        ready.append(True)
        return _two_action_graph()
    def forbidden(*args, **kwargs):
        raise AssertionError("expired campaign must not launch proof")
    monkeypatch.setattr(executor_module.time, "perf_counter",
        lambda: clock() + (16 if ready and reason == "maximum_seconds" else 0))
    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "prove", forbidden)
    result = _executor(rules).execute(_matching_scenario(complete_candidate), _proof(),
        candidate_compiler=candidate, graph_compiler=graph, maximum_seconds=15,
        cancel_check=lambda: bool(ready) and reason == "cancelled")
    assert result.status == "incomplete"
    assert result.candidate is complete_candidate
    assert len(graph_calls) == int(stage == "proof")
    receipt = result.execution_receipt["incompleteCompilation"]
    assert receipt["stage"] == stage
    assert receipt["exhaustionReason"] == reason
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.parametrize("stage", ["graph", "proof"])
@pytest.mark.parametrize("reason", ["maximum_seconds", "maximum_nodes", "maximum_depth", "cancelled"])
def test_typed_graph_stop_preserves_cause(rules, complete_candidate, monkeypatch, stage, reason):
    stopped = executor_module.ExactNormalizationStopped(reason, evidence={"sourceStage": "native_test"})

    def stop(*args, **kwargs):
        raise stopped

    if stage == "proof":
        monkeypatch.setattr(OrToolsScenarioGraphProofV1, "prove", stop)
    result = _executor(rules).execute(
        _matching_scenario(complete_candidate), _proof(),
        candidate_compiler=lambda *args, **kwargs: OrToolsCandidateAttemptV1("feasible", complete_candidate),
        graph_compiler=stop if stage == "graph" else lambda *args, **kwargs: _two_action_graph(),
        maximum_seconds=15,
    )
    assert result.status == "incomplete"
    receipt = result.execution_receipt["incompleteCompilation"]
    assert receipt["stage"] == stage
    assert receipt["exhaustionReason"] == reason
    assert receipt["evidence"]["causeReceipt"] == stopped.partial_receipt
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


def test_default_complete_graph_receives_live_cancellation(root, rules, monkeypatch):
    from baloot_platform.research.ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
    prefix, _ = _quality_continuation_input(root, "exact")
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"))
    entered = []
    original = OrToolsCompleteStateGraphCompilerV1.compile
    def compile_then_cancel(*args, **kwargs):
        assert callable(kwargs["cancelled"])
        entered.append(True)
        result = original(*args, **kwargs)
        assert not result.complete
        assert result.receipt.exhaustion_reason == "cancelled"
        return result
    monkeypatch.setattr(OrToolsCompleteStateGraphCompilerV1, "compile", compile_then_cancel)
    result = _executor(rules).execute(scenario, _proof(), maximum_seconds=30,
        maximum_proof_nodes=1000, maximum_proof_depth=8,
        cancel_check=lambda: bool(entered))
    assert entered
    assert result.status == "incomplete"
    assert result.candidate is not None
    assert result.proof is None
    receipt = result.execution_receipt["incompleteCompilation"]
    assert receipt["stage"] == "graph"
    assert receipt["exhaustionReason"] == "cancelled"


def test_graph_budget_or_compilation_refusal_is_incomplete(rules, complete_candidate):
    scenario = _matching_scenario(complete_candidate)

    def candidate_compiler(*_args, **_kwargs):
        return OrToolsCandidateAttemptV1("feasible", complete_candidate)

    def refused_graph(*_args, **_kwargs):
        raise RuntimeError("governed graph budget exceeded")

    result = _executor(rules).execute(
        scenario,
        _proof(),
        candidate_compiler=candidate_compiler,
        graph_compiler=refused_graph,
        maximum_seconds=15,
    )
    assert result.status == "incomplete"
    assert result.candidate is complete_candidate
    assert result.proof is None


def test_exact_trace_prefix_normalizes_without_runtime_engine(
    monkeypatch, root: Path, rules
):
    def unexpected_tail_plan(*_args, **_kwargs):
        raise AssertionError("exact trace-prefix execution planned a search tail")

    monkeypatch.setattr(TailFirstStagePlannerV1, "plan", unexpected_tail_plan)
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    origin = {
        "kind": "exact_trace_prefix",
        "exactTracePrefix": fixture["exactTracePrefix"],
    }
    scenario = _scenario(
        origin=origin,
        constraints=_leaf(
            "PRED-EXACT-PHASE",
            "state.phase",
            {"state": "playing"},
        ),
        outcome=_leaf(
            "PRED-EXACT-ACTOR",
            "state.next_actor",
            {"seat": fixture["expected"]["actingSeat"]},
        ),
    )
    original_normalize = OrToolsExactStateNormalizerV1.normalize
    normalization_calls = 0

    def counted_normalize(self, *args, **kwargs):
        nonlocal normalization_calls
        normalization_calls += 1
        return original_normalize(self, *args, **kwargs)

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", counted_normalize)
    result = _executor(rules).execute(
        scenario,
        _proof(),
        maximum_seconds=15,
    )
    assert result.status == "feasible"
    assert result.candidate.origin == "exact_trace_prefix"
    assert result.candidate.exact_state_id.startswith("STATE-")
    assert result.predicate_evaluation.satisfied
    assert "normalizationReceipt" in result.candidate.trace

    exact_proof = ScenarioProofRequestV1.build(fixture["proof"])
    proved = _executor(rules).execute(
        scenario,
        exact_proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_seconds=15,
    )
    assert proved.status == "qualified"
    assert normalization_calls == 3
    assert proved.execution_receipt["completedCandidates"] == 1
    timings = proved.execution_receipt["stageTimings"]
    assert set(timings) == {
        "construction",
        "playingQuality",
        "qualityAuction",
        "qualityProjectPlay",
        "tailConstruction",
        "tailModelBuild",
        "tailModelConfigure",
        "tailSolver",
        "prefixReachability",
        "proofBoundary",
        "proofBoundaryModelBuild",
        "proofBoundarySolver",
        "prefixLift",
        "exactNormalization",
        "finalNormalization",
        "graph",
        "proof",
        "qualification",
        "total",
    }
    assert all(value >= 0 for value in timings.values())
    assert timings["total"] >= sum(
        value for stage, value in timings.items()
        if stage not in {"total", "qualityAuction", "qualityProjectPlay"}
    )
    assert proved.proof["winningRootActionCount"] == 1
    assert proved.proof["selectedAction"] == "CA"


@pytest.mark.parametrize(
    ("claim", "success_threshold", "line_universe"),
    (
        ("winning_root_action_count", 1, None),
        ("outcome_forced", -1, None),
        ("winning_continuation_line_count", 1, "policy_conditioned"),
    ),
)
def test_exact_policy_conditioned_templates_materialize_and_complete(
    root: Path,
    rules,
    claim: str,
    success_threshold: int,
    line_universe: str | None,
):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    scenario = _scenario(
        origin={
            "kind": "exact_trace_prefix",
            "exactTracePrefix": fixture["exactTracePrefix"],
        },
        constraints=_leaf("PRED-POLICY-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf(
            "PRED-POLICY-ACTOR",
            "state.next_actor",
            {"seat": fixture["expected"]["actingSeat"]},
        ),
    )
    proof_payload = {
        "schemaVersion": 1,
        "mode": "policy_conditioned",
        "visibility": "complete_state",
        "strategyClass": "frozen_policy",
        "claim": claim,
        "targetTeam": "west_east",
        "utility": "round_win",
        "successThreshold": success_threshold,
        "selectedAction": "CA",
        "policies": _policy_templates(),
        "budgetProfile": "late_game_exact_v1",
    }
    if line_universe is not None:
        proof_payload["lineUniverse"] = line_universe
    proof = ScenarioProofRequestV1.build(proof_payload)

    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_seconds=15,
    )

    assert result.status == "qualified"
    assert result.proof["completionStatus"] == "policy_complete"
    assert type(result.proof["winningRootActionCount"]) is int
    if line_universe is not None:
        assert result.proof["continuationLines"]["lineUniverse"] == line_universe
        assert result.proof["continuationLines"]["exactCount"] in {"0", "1"}
    identities = result.proof["policyIdentities"]
    assert all(
        identity["policyId"] == "fixed_action_map_v1"
        for identity in identities.values()
    )
    assert all("materializedHash" in identity for identity in identities.values())
    candidate = result.candidate
    assert candidate is not None
    source_prefix = candidate.trace["exactTracePrefix"]
    complete_prefix = candidate.trace["completeExactTracePrefix"]
    assert len(source_prefix["playActions"]) == 23
    assert len(complete_prefix["playActions"]) == 32
    assert [
        LegalPlayActionV1(
            Card.parse(item["card"]),
            bool(item.get("ekka", False)),
        ).code
        for item in complete_prefix["playActions"][23:]
    ] == result.proof["principalVariation"]
    assert candidate.trace["completeExactNormalization"]["phase"] == "scored"
    assert (
        len(candidate.trace["completeExactNormalization"]["state"]["completedTricks"])
        == 8
    )
    continuation = candidate.trace["policyContinuation"]
    graph = result.semantic_artifact_documents()["proofGraph"]
    assert continuation["proofOutcomeHash"] == result.proof["sha256"]
    assert continuation["graphHash"] == canonical_sha256(
        {key: value for key, value in graph.items() if key != "sha256"}
    )
    assert continuation["terminalScore"] == candidate.trace["score"]


def test_policy_play_qualification_uses_pv_terminal_trace_and_decision_state(
    root: Path,
    rules,
):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    scenario = _scenario(
        origin={
            "kind": "exact_trace_prefix",
            "exactTracePrefix": fixture["exactTracePrefix"],
        },
        constraints={
            "op": "all",
            "children": [
                _leaf("PRED-POLICY-DECISION", "state.phase", {"state": "playing"}),
                _leaf(
                    "PRED-POLICY-DECISION-TRICK",
                    "play.trick_index",
                    {"minimum": 5, "maximum": 5},
                ),
                _leaf(
                    "PRED-POLICY-PV-ROOT",
                    "play.action_matches",
                    {
                        "trickIndex": 5,
                        "position": 3,
                        "actingSeat": "west",
                        "selector": {
                            "ranks": ["ace"],
                            "suits": ["clubs"],
                            "owner": {"relation": "actor"},
                        },
                    },
                ),
            ],
        },
        outcome=_leaf(
            "PRED-POLICY-PV-OUTCOME",
            "outcome.team_round",
            {"team": "west_east", "status": "won"},
        ),
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 1,
            "selectedAction": "CA",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_seconds=15,
    )

    assert result.status == "qualified"
    assert result.candidate is not None
    assert result.candidate.state["phase"] == "playing"
    assert result.candidate.trace["exactNormalization"]["legalActions"]
    assert result.candidate.trace["completePlay"]["decisionState"]["prefixLength"] == 32
    assert result.candidate.trace["score"]["roundWinner"] == "west_east"
    evidence = {row["kind"]: row for row in result.predicate_evaluation.evidence}
    assert evidence["play.action_matches"]["observed"]["action"]["card"]["code"] == "CA"
    assert evidence["outcome.team_round"]["observed"]["roundWinner"] == ("west_east")


def test_search_construction_witness_is_preserved_while_pv_qualifies(
    rules,
    complete_candidate: OrToolsNeutralCandidateV1,
):
    trace = copy.deepcopy(dict(complete_candidate.trace))
    trace["play"]["plays"][24]["ekka"] = not bool(
        trace["play"]["plays"][24].get("ekka", False)
    )
    mismatched = OrToolsNeutralCandidateV1(
        complete_candidate.origin,
        trace,
        complete_candidate.state,
        complete_candidate.no_good_vector,
        complete_candidate.stage_receipts,
        complete_candidate.exact_state_id,
    )
    decision = complete_candidate.trace["play"]["decisionState"]
    selected_row = decision["legalPlayActions"][0]
    selected = LegalPlayActionV1(
        Card.parse(selected_row["card"]),
        bool(selected_row.get("ekka", False)),
    )
    scenario = _scenario(
        constraints={
            "op": "all",
            "children": [
                _leaf("PRED-SEARCH-POLICY-STATE", "state.phase", {"state": "playing"}),
                _leaf(
                    "PRED-SEARCH-POLICY-ROOT",
                    "play.action_matches",
                    {
                        "trickIndex": 6,
                        "position": 0,
                        "actingSeat": decision["nextActor"],
                        "selector": {
                            "ranks": [selected.card.rank.name.lower()],
                            "suits": [selected.card.suit.name.lower()],
                            "owner": {"relation": "actor"},
                        },
                    },
                ),
            ],
        },
        outcome=_leaf(
            "PRED-SEARCH-POLICY-OUTCOME",
            "outcome.round_margin",
            {"team": "north_south", "minimum": -100000, "maximum": 100000},
        ),
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": -1,
            "selectedAction": selected.code,
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", mismatched
        ),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_candidates=1,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.candidate is not None
    retained = result.candidate.trace
    assert retained["play"] == mismatched.trace["play"]
    assert retained["constructionScore"] == mismatched.trace["score"]
    source_count = len(retained["exactTracePrefix"]["playActions"])
    assert [
        LegalPlayActionV1(Card.parse(item["card"]), bool(item.get("ekka", False))).code
        for item in retained["completeExactTracePrefix"]["playActions"][source_count:]
    ] == result.proof["principalVariation"]
    assert retained["completePlay"]["plays"] != retained["play"]["plays"]
    assert retained["policyContinuation"]["sourceConstructionPlayHash"] == (
        canonical_sha256(retained["play"])
    )
    evidence = {row["kind"]: row for row in result.predicate_evaluation.evidence}
    assert (
        evidence["play.action_matches"]["observed"]["action"]["card"]["code"]
        == selected.card.code
    )
    expected_margin = (
        retained["score"]["gamePoints"][0] - retained["score"]["gamePoints"][1]
    )
    assert evidence["outcome.round_margin"]["observed"]["value"] == expected_margin


def test_exact_complete_uses_pv_trace_and_rejects_bound_artifact_substitution(
    monkeypatch,
    root: Path,
    rules,
):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    two_ace_prefix = copy.deepcopy(fixture["exactTracePrefix"])
    deck = two_ace_prefix["deckOrder"]
    seven_index = deck.index(Card.parse("D7").id)
    ace_index = deck.index(Card.parse("SA").id)
    deck[seven_index], deck[ace_index] = deck[ace_index], deck[seven_index]
    two_ace_prefix.pop("sha256", None)
    seed_scenario = _scenario(
        origin={"kind": "exact_trace_prefix", "exactTracePrefix": two_ace_prefix},
        decision_point={
            "kind": "play",
            "selector": {
                "trickIndex": 5,
                "position": 3,
                "actingSeat": "west",
                "remainingCards": 3,
            },
        },
        constraints=_leaf(
            "PRED-EXACT-DIVERGENT-SEED-STATE",
            "state.phase",
            {"state": "playing"},
        ),
        outcome=_leaf(
            "PRED-EXACT-DIVERGENT-SEED-OUTCOME",
            "outcome.team_round",
            {"team": "north_south", "status": "won"},
            "report",
        ),
    )
    losing_construction = _executor(rules).execute(
        seed_scenario,
        _proof(selected_action="SA", target_team="west_east"),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_seconds=30,
    )
    assert losing_construction.status == "qualified"
    seed_candidate = losing_construction.candidate
    assert seed_candidate is not None
    assert seed_candidate.trace["score"]["roundWinner"] == "north_south"

    construction_trace = copy.deepcopy(dict(seed_candidate.trace))
    construction_trace.update(
        {
            "deckOrder": copy.deepcopy(two_ace_prefix["deckOrder"]),
            "dealer": two_ace_prefix["dealer"],
            "roundNumber": two_ace_prefix.get("roundNumber", 1),
            "matchScore": copy.deepcopy(two_ace_prefix.get("matchScore", [0, 0])),
            "buyer": "north",
            "contract": "sun",
            "trumpSuit": None,
        }
    )
    construction_play = copy.deepcopy(dict(construction_trace["completePlay"]))
    construction_play["decisionState"] = copy.deepcopy(
        dict(construction_trace["play"])["decisionState"]
    )
    construction_trace["play"] = construction_play
    construction_trace["auction"] = {
        **dict(construction_trace["auction"]),
        "completion": {
            "actions": copy.deepcopy(two_ace_prefix["auctionActions"]),
            "buyer": "north",
            "contract": "sun",
            "trumpSuit": None,
            "purchaseRound": 1,
        },
    }
    for key in (
        "completePlay",
        "completeExactTracePrefix",
        "completeExactNormalization",
        "proofContinuation",
    ):
        construction_trace.pop(key, None)
    construction = OrToolsNeutralCandidateV1(
        "search",
        construction_trace,
        seed_candidate.state,
        seed_candidate.no_good_vector,
        {
            key: value
            for key, value in seed_candidate.stage_receipts.items()
            if key not in {"proof", "tactics", "proofContinuation"}
        },
        seed_candidate.exact_state_id,
    )
    assert construction.trace["play"]["plays"][23]["card"] == "SA"

    exact_scenario = _scenario(
        decision_point={
            "kind": "play",
            "selector": {
                "trickIndex": 5,
                "position": 3,
                "actingSeat": "west",
                "remainingCards": 3,
            },
        },
        constraints={
            "op": "all",
            "children": [
                _leaf(
                    "PRED-EXACT-DIVERGENT-STATE",
                    "state.phase",
                    {"state": "playing"},
                ),
                _leaf(
                    "PRED-EXACT-DIVERGENT-ROOT",
                    "play.action_matches",
                    {
                        "trickIndex": 5,
                        "position": 3,
                        "actingSeat": "west",
                        "selector": {
                            "ranks": ["ace"],
                            "suits": ["clubs"],
                            "owner": {"relation": "actor"},
                        },
                    },
                ),
                _leaf(
                    "PRED-EXACT-DIVERGENT-SEVEN",
                    "play.trick_winner_matches",
                    {
                        "trickIndex": 6,
                        "team": "west_east",
                        "selector": {
                            "ranks": ["eight"],
                            "suits": ["clubs"],
                            "owner": {"relation": "team", "team": "west_east"},
                        },
                    },
                ),
            ],
        },
        outcome=_leaf(
            "PRED-EXACT-DIVERGENT-OUTCOME",
            "outcome.team_round",
            {"team": "west_east", "status": "won"},
        ),
    )
    captured: dict[str, object] = {}
    original_prove = OrToolsScenarioGraphProofV1.prove

    def capture(self, graph, request, **kwargs):
        outcome = original_prove(self, graph, request, **kwargs)
        captured.update(graph=graph, proof=outcome)
        return outcome

    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "prove", capture)
    result = _executor(rules).execute(
        exact_scenario,
        _proof(selected_action="CA", target_team="west_east"),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", construction
        ),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_candidates=1,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    candidate = result.candidate
    assert candidate is not None
    retained = candidate.trace
    assert retained["play"] == construction.trace["play"]
    assert retained["constructionScore"] == construction.trace["score"]
    assert retained["play"]["plays"][23]["card"] == "SA"
    assert retained["completePlay"]["plays"][23]["card"] == "CA"
    source_count = len(retained["exactTracePrefix"]["playActions"])
    assert source_count == 23
    assert [
        LegalPlayActionV1(Card.parse(item["card"]), bool(item.get("ekka", False))).code
        for item in retained["completeExactTracePrefix"]["playActions"][source_count:]
    ] == result.proof["principalVariation"]
    assert len(retained["completeExactTracePrefix"]["playActions"]) == 32
    assert retained["completePlay"]["tricks"][6]["winningCard"] == "C8"
    assert retained["score"]["roundWinner"] == "west_east"
    continuation = retained["proofContinuation"]
    assert continuation["sourceConstructionPlayHash"] == canonical_sha256(
        retained["play"]
    )
    assert continuation["sourceConstructionScoreHash"] == canonical_sha256(
        retained["constructionScore"]
    )
    evidence = {row["predicateId"]: row for row in result.predicate_evaluation.evidence}
    assert (
        evidence["PRED-EXACT-DIVERGENT-ROOT"]["observed"]["action"]["card"]["code"]
        == "CA"
    )
    assert (
        evidence["PRED-EXACT-DIVERGENT-SEVEN"]["observed"]["winner"]["card"]["code"]
        == "C8"
    )
    assert evidence["PRED-EXACT-DIVERGENT-OUTCOME"]["observed"]["roundWinner"] == (
        "west_east"
    )

    def rebound(
        trace: dict,
        receipts: dict[str, OrToolsResearchStageReceiptV1] | None = None,
    ) -> OrToolsNeutralCandidateV1:
        return OrToolsNeutralCandidateV1(
            candidate.origin,
            trace,
            candidate.state,
            candidate.no_good_vector,
            receipts or dict(candidate.stage_receipts),
            candidate.exact_state_id,
        )

    for field, replacement in (
        ("proofOutcomeHash", "0" * 64),
        ("principalVariation", ["SA", *result.proof["principalVariation"][1:]]),
    ):
        trace = copy.deepcopy(dict(retained))
        trace["proofContinuation"][field] = replacement
        core = dict(trace["proofContinuation"])
        core.pop("sha256")
        trace["proofContinuation"]["sha256"] = canonical_sha256(core)
        with pytest.raises(ValueError, match="proof binding mismatch"):
            _validate_play_policy_continuation_binding_v1(
                rebound(trace),
                graph=captured["graph"],
                proof=captured["proof"],
            )

    trace = copy.deepcopy(dict(retained))
    changed_prefix = copy.deepcopy(dict(trace["completeExactTracePrefix"]))
    changed_prefix.pop("sha256")
    changed_prefix["playActions"][23], changed_prefix["playActions"][24] = (
        changed_prefix["playActions"][24],
        changed_prefix["playActions"][23],
    )
    changed_prefix = ExactTracePrefixV1.build(changed_prefix).to_dict()
    trace["completeExactTracePrefix"] = changed_prefix
    trace["proofContinuation"]["completeExactTracePrefixHash"] = changed_prefix[
        "sha256"
    ]
    trace["proofContinuation"]["completeExactTracePrefixDocumentHash"] = (
        canonical_sha256(changed_prefix)
    )
    core = dict(trace["proofContinuation"])
    core.pop("sha256")
    trace["proofContinuation"]["sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="suffix mismatch"):
        _validate_play_policy_continuation_binding_v1(
            rebound(trace),
            graph=captured["graph"],
            proof=captured["proof"],
        )

    receipts = dict(candidate.stage_receipts)
    play_receipt = receipts["playPrefix"]
    receipts["playPrefix"] = OrToolsResearchStageReceiptV1(
        play_receipt.stage,
        play_receipt.model_hash,
        play_receipt.evidence_hash,
        "0" * 64,
        play_receipt.solver_status,
        play_receipt.branches,
        play_receipt.conflicts,
    )
    with pytest.raises(ValueError, match="stage receipt mismatch"):
        _validate_play_policy_continuation_binding_v1(
            rebound(copy.deepcopy(dict(retained)), receipts),
            graph=captured["graph"],
            proof=captured["proof"],
        )


def test_play_policy_continuation_hash_substitution_fails_closed(
    monkeypatch,
    root: Path,
    rules,
):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    scenario = _scenario(
        origin={
            "kind": "exact_trace_prefix",
            "exactTracePrefix": fixture["exactTracePrefix"],
        },
        constraints=_leaf("PRED-BINDING-STATE", "state.phase", {"state": "playing"}),
        outcome=_leaf(
            "PRED-BINDING-REPORT",
            "state.next_actor",
            {"seat": fixture["expected"]["actingSeat"]},
            "report",
        ),
    )
    proof_request = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 1,
            "selectedAction": "CA",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )
    captured: dict[str, object] = {}
    original_prove = OrToolsScenarioGraphProofV1.prove

    def capture(self, graph, request, **kwargs):
        outcome = original_prove(self, graph, request, **kwargs)
        captured.update(graph=graph, proof=outcome)
        return outcome

    monkeypatch.setattr(OrToolsScenarioGraphProofV1, "prove", capture)
    result = _executor(rules).execute(
        scenario,
        proof_request,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_seconds=15,
    )
    assert result.status == "qualified"
    candidate = result.candidate
    assert candidate is not None
    trace = copy.deepcopy(dict(candidate.trace))
    trace["policyContinuation"]["proofOutcomeHash"] = "0" * 64
    substituted_core = dict(trace["policyContinuation"])
    substituted_core.pop("sha256")
    trace["policyContinuation"]["sha256"] = canonical_sha256(substituted_core)
    tampered = OrToolsNeutralCandidateV1(
        candidate.origin,
        trace,
        candidate.state,
        candidate.no_good_vector,
        candidate.stage_receipts,
        candidate.exact_state_id,
    )
    with pytest.raises(ValueError, match="proof binding mismatch"):
        _validate_play_policy_continuation_binding_v1(
            tampered,
            graph=captured["graph"],
            proof=captured["proof"],
        )


@pytest.mark.real_clock
@pytest.mark.parametrize("stage", ["graph", "proof"])
@pytest.mark.parametrize("reason", ["maximum_seconds", "cancelled"])
def test_information_handoff_never_launches_after_stop(root, rules, monkeypatch, stage, reason):
    prefix, proof = _quality_continuation_input(root, "exact")
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"))
    executor = _executor(rules)
    original = executor._candidate_from_normalized_exact
    clock = executor_module.time.perf_counter
    ready = []
    calls = []
    def candidate(*args, **kwargs):
        result = original(*args, **kwargs)
        if stage == "graph":
            ready.append(True)
        return result
    def compile_graph(*args, **kwargs):
        calls.append(True)
        assert stage == "proof"
        ready.append(True)
        return SimpleNamespace()
    def forbidden(*args, **kwargs):
        raise AssertionError("stopped campaign must not launch information proof")
    monkeypatch.setattr(executor, "_candidate_from_normalized_exact", candidate)
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", compile_graph)
    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", forbidden)
    monkeypatch.setattr(executor_module.time, "perf_counter",
        lambda: clock() + (31 if ready and reason == "maximum_seconds" else 0))
    result = executor.execute(scenario, proof, maximum_seconds=30,
        maximum_proof_nodes=1000, maximum_proof_depth=8, maximum_hidden_worlds=1,
        cancel_check=lambda: bool(ready) and reason == "cancelled")
    assert result.status == "incomplete"
    assert result.candidate is not None
    assert result.proof is None
    assert len(calls) == int(stage == "proof")
    receipt = result.execution_receipt["incompleteCompilation"]
    assert receipt["stage"] == f"information_{stage}_handoff"
    assert receipt["exhaustionReason"] == reason
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})


@pytest.mark.real_clock
def test_exact_actor_information_set_proof_is_exhaustive_and_observer_safe(
    root: Path, rules, monkeypatch
):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = copy.deepcopy(fixture["exactTracePrefix"])
    prefix["playActions"].extend(
        (
            {"actor": "west", "card": "CA"},
            {"actor": "east", "card": "C8"},
            {"actor": "north", "card": "C7"},
            {"actor": "west", "card": "D7"},
            {"actor": "south", "card": "S7"},
        )
    )
    prefix["stateAssertions"] = {}
    scenario = _scenario(
        origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        constraints=_leaf("PRED-INFO-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-INFO-ACTOR", "state.next_actor", {"seat": "east"}),
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 1,
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=4,
        maximum_hidden_worlds=64,
        maximum_seconds=15,
    )

    assert result.status == "qualified"
    assert result.proof["completionStatus"] == "exact_complete"
    assert result.proof["winningRootActionCount"] == 1
    assert result.proof["hiddenWorlds"]["count"] == 2
    assert result.proof["hiddenWorlds"]["complete"] is True
    assert result.proof["proofEngine"]["id"] == (
        "ortools_actor_information_set_cegis_v1"
    )
    documents = result.semantic_artifact_documents()
    encoded = json.dumps(documents["proofGraph"], sort_keys=True)
    assert '"hands"' not in encoded
    assert '"initialHands"' not in encoded

    incomplete = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=4,
        maximum_hidden_worlds=1,
        maximum_seconds=15,
    )
    assert incomplete.status == "incomplete"
    assert incomplete.proof is None
    partial = incomplete.execution_receipt["incompleteCompilation"]
    assert partial["exhaustionReason"] == "maximum_hidden_worlds"
    assert partial["evidence"]["acceptedWorlds"] == 2
    assert partial["sha256"] == canonical_sha256(
        {key: value for key, value in partial.items() if key != "sha256"}
    )

    normalizing = []
    original_normalize = OrToolsExactStateNormalizerV1.normalize
    def normalize_then_cancel(*args, **kwargs):
        normalizing.append(True)
        return original_normalize(*args, **kwargs)
    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", normalize_then_cancel)

    cancelled = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=4,
        maximum_hidden_worlds=64,
        maximum_seconds=15,
        cancel_check=lambda: bool(normalizing),
    )
    assert cancelled.status == "incomplete"
    assert cancelled.execution_receipt["incompleteCompilation"]["stage"] == "exactNormalization"
    assert (
        cancelled.execution_receipt["incompleteCompilation"]["exhaustionReason"]
        == "cancelled"
    )


def test_native_actor_observer_scope_is_owned_by_real_proof_and_candidate(root, rules, monkeypatch):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix, proof_request = _quality_continuation_input(root, "exact")
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))
    raw = prefix.to_dict()
    raw.pop("sha256")
    raw["playActions"].append({"card": fixture["expected"]["principalVariation"][4], "ekka": False})
    prefix = ExactTracePrefixV1.build(raw)
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-INFO-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-ACTOR-SCOPE", "state.observer_visibility",
            {"observer": "east", "classification": "legal_deduction"}))
    captured = {}
    compile_original = executor_module.compile_observed_decision_v1
    prove_original = executor_module.prove_actor_information_set_v1
    def compile_once(*args, **kwargs):
        assert "graph" not in captured
        compilation = compile_original(*args, **kwargs)
        captured["graph"] = compilation.graph
        return compilation
    def prove(*args, **kwargs):
        outcome = prove_original(*args, **kwargs)
        captured["proof"] = outcome
        return outcome
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", compile_once)
    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", prove)
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-ACTOR-SCOPE", bundle_hash="a" * 64)
    result = executor.execute(scenario, proof_request, maximum_proof_nodes=100_000,
        maximum_proof_depth=4, maximum_hidden_worlds=64, maximum_seconds=30)
    assert result.status == "qualified", result.execution_receipt
    graph, proof = captured["graph"], captured["proof"]
    scope = graph.observer_scope
    assert scope.visibility_scope == "actor_information_set"
    assert scope.hidden_world_count == result.proof["hiddenWorlds"]["count"] == 2
    evidence = next(row for row in result.predicate_evaluation.evidence if row["kind"] == "state.observer_visibility")
    assert evidence["satisfied"] is True
    assert evidence["observed"] == scope.to_safe_dict()
    encoded = json.dumps(evidence)
    assert all(key not in encoded for key in ('"hands"', '"worldHashes"', '"worldStates"'))
    assert result.proof["identities"]["graphHash"] == graph.to_dict()["sha256"]
    assert graph.model_document["observerScopeBinding"]["scope"] == scope.to_safe_dict()
    provider = DefaultStageContextCompilerV1(rules, required_stages=("state",),
        native_cp_sat_authority=True, actor_observer_scope_required=True)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"})
    search_candidate = executor_module._rebind_search_candidate_to_decision(
        replace(result.candidate, origin="search"), prefix, normalized,
        preserve_construction_outcome=True, native_cp_sat=True)
    search_context = provider(search_candidate, proof, information_graph=graph)["state"].context
    assert search_context.observer_scope == scope
    for candidate, supplied_graph, supplied_proof in (
        (replace(result.candidate, exact_state_id="STATE-OTHER"), graph, proof),
        (result.candidate, replace(graph, hidden_worlds={**graph.hidden_worlds, "complete": False}), proof),
        (result.candidate, replace(graph, observer_scope=None), proof),
        (result.candidate, graph, replace(proof, completion_status="proof_incomplete")),
        (result.candidate, graph, replace(proof, graph_hash="0" * 64)),
    ):
        with pytest.raises(ValueError, match="observer scope"):
            provider(candidate, supplied_proof, information_graph=supplied_graph)

    captured.clear()
    failed_request = proof_request.to_dict()
    failed_request.pop("sha256")
    failed_request.update(claim="outcome_reachable", successThreshold=1000)
    failed = executor.execute(scenario, ScenarioProofRequestV1.build(failed_request),
        maximum_proof_nodes=100_000, maximum_proof_depth=4, maximum_hidden_worlds=64,
        maximum_seconds=30)
    assert failed.status == "proof_failed", failed.execution_receipt
    assert failed.proof["completionStatus"] == "proof_failed"
    assert failed.predicate_evaluation.status == "satisfied"


def test_native_complete_state_small_card_gain_uses_real_legal_prefix(root):
    from baloot_platform.tactics import TacticalDetectorProfileV1

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix, _ = _quality_continuation_input(root, "exact")
    raw = prefix.to_dict()
    raw.pop("sha256")
    # Saving the Ace feeds the partner's last diamond winner.
    exchange = {Card.parse("C9").id: Card.parse("D9").id,
                Card.parse("D9").id: Card.parse("C9").id}
    raw["deckOrder"] = [exchange.get(card, card) for card in raw["deckOrder"]]
    for row in raw["playActions"]:
        card = Card.parse(row["card"])
        row["card"] = Card(exchange.get(card.id, card.id)).code
    scenario = _scenario(
        origin={"kind": "exact_trace_prefix", "exactTracePrefix": raw},
        constraints=_leaf("PRED-SMALL-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-SMALL-GAIN", "tactic.verified", {
            "kind": "small_card_big_gain", "minimumScore": 100,
            "minimumEvidence": 1, "causalRequired": False,
            "actingSeat": "south", "targetTeam": "north_south"}),
    )
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "strategyClass": "pure_deterministic", "claim": "outcome_reachable",
        "targetTeam": "north_south", "utility": "awarded_point_margin",
        "successThreshold": 10, "selectedAction": "S7", "budgetProfile": "test_exact",
    })
    result = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-NATIVE-SMALL-GAIN", bundle_hash="a" * 64,
        tactical_detector_profile=TacticalDetectorProfileV1.defaults(),
    ).execute(scenario, request, maximum_seconds=30,
              maximum_proof_nodes=100_000, maximum_proof_depth=8)

    assert result.status == "qualified", result.execution_receipt
    assert result.research_authority["backendId"] == "ortools_cp_sat_v3"
    assert result.proof["completionStatus"] == "exact_complete"
    values = {row["action"]: row["value"] for row in result.proof["rootActions"]}
    assert values["S7"] > values["SA"]
    evidence = next(row for row in result.predicate_evaluation.evidence
                    if row["predicateId"] == "PRED-SMALL-GAIN")
    assert evidence["satisfied"] is True
    assert evidence["evidenceExtractor"] == "cp_sat.native_proof_same_solve_v1"
    certificate = evidence["observed"]
    assert certificate == result.proof["nativeQualification"]["PRED-SMALL-GAIN"]
    assert certificate["observed"]["qualificationSource"] == "native_cp_sat_root_value_variables"
    assert certificate["observed"]["rootValueVariables"]
    assert certificate["observed"]["verified"] is True
    assert certificate["observed"]["causalProven"] is False


def test_native_actor_root_predicates_use_owned_exact_certificates(root, monkeypatch):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    detector_profile = TacticalDetectorProfileV1.defaults()
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix, _request = _quality_continuation_input(root, "exact")
    raw = prefix.to_dict()
    raw.pop("sha256")
    # Existing partner-feed fixture: saving the Ace feeds the last diamond winner.
    exchange = {Card.parse("C9").id: Card.parse("D9").id,
                Card.parse("D9").id: Card.parse("C9").id}
    raw["deckOrder"] = [exchange.get(card, card) for card in raw["deckOrder"]]
    for row in raw["playActions"]:
        card = Card.parse(row["card"])
        row["card"] = Card(exchange.get(card.id, card.id)).code
    prefix = ExactTracePrefixV1.build(raw)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": "north_south", "utility": "awarded_point_margin", "successThreshold": 10,
        "budgetProfile": "actor_information_set_exact_v1",
    })
    root_claims = [
        _leaf("PRED-ROOT-COUNT", "proof.winning_root_action_count", {"minimum": 1, "maximum": 1}),
        _leaf("PRED-ROOT-RANK", "proof.winning_root_card_rank_count", {
            "rank": ["seven"], "minimum": 1, "maximum": 1}),
        _leaf("PRED-ROOT-SELECTOR", "proof.winning_root_action_matches", {
            "selector": {"cards": [Card.parse("S7").id], "owner": {"relation": "seat", "seat": "south"}}}),
        _leaf("PRED-ROOT-APPLICABLE", "tactic.applicable", {
            "kind": "only_winning_move", "applicable": True, "observer": "south"}),
        _leaf("PRED-ROOT-VERIFIED", "tactic.verified", {
            "kind": "only_winning_move", "minimumScore": 100, "minimumEvidence": 1,
            "causalRequired": True, "actingSeat": "south", "targetTeam": "north_south"}),
        _leaf("PRED-ROOT-SMALL-APPLICABLE", "tactic.applicable", {
            "kind": "small_card_big_gain", "applicable": True, "actingSeat": "south"}),
        _leaf("PRED-ROOT-SMALL-VERIFIED", "tactic.verified", {
            "kind": "small_card_big_gain", "minimumScore": 100, "minimumEvidence": 1,
            "causalRequired": False, "actingSeat": "south", "targetTeam": "north_south"}),
    ]
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-ROOT-PHASE", "state.phase", {"state": "playing"}),
        outcome={"op": "all", "children": root_claims})
    captured = {}
    original_compile = executor_module.compile_observed_decision_v1
    original_prove = executor_module.prove_actor_information_set_v1
    def compile_graph(*args, **kwargs):
        result = original_compile(*args, **kwargs)
        captured["graph"] = result.graph
        return result
    def prove(*args, **kwargs):
        result = original_prove(*args, **kwargs)
        captured["proof"] = result
        return result
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", compile_graph)
    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", prove)
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-ACTOR-ROOT", bundle_hash="a" * 64,
        tactical_detector_profile=detector_profile)
    limits = dict(maximum_proof_nodes=100_000, maximum_proof_depth=8,
        maximum_hidden_worlds=64, maximum_seconds=30)
    result = executor.execute(scenario, request, **limits)
    assert result.status == "qualified", json.dumps({"receipt": result.execution_receipt,
        "proof": captured["proof"].to_dict(),
        "evaluation": result.predicate_evaluation.to_dict() if result.predicate_evaluation else None}, sort_keys=True)
    assert result.proof["winningRootActionCount"] == 1
    assert result.proof["hiddenWorlds"]["complete"] is True
    assert result.proof["hiddenWorlds"]["count"] == 2
    assert {row["action"]: row["value"] for row in result.proof["rootActions"]} == {"S7": 14, "SA": 8}
    proof, graph = captured["proof"], captured["graph"]
    observed = {row["predicateId"]: row["observed"] for row in result.predicate_evaluation.evidence}
    assert observed["PRED-ROOT-RANK"]["matchingCount"] == 1
    assert "matchingActions" not in observed["PRED-ROOT-RANK"]
    assert observed["PRED-ROOT-SELECTOR"]["winningActions"][0]["card"]["code"] == "S7"
    tactic = observed["PRED-ROOT-VERIFIED"]
    assert tactic["verified"] is True and tactic["causalProven"] is True
    assert tactic["qualificationSource"] == "native_cp_sat_root_win_variables"
    assert tactic["winningVariables"] == [row["winningVariable"]
        for row in proof.model_document()["rootActionVariables"]]
    assert observed["PRED-ROOT-SMALL-VERIFIED"]["verified"] is True
    assert observed["PRED-ROOT-SMALL-VERIFIED"]["causalProven"] is False
    assert observed["PRED-ROOT-SMALL-VERIFIED"]["causalProofStatus"] == "missing_native_delayed_payoff_witness"
    provider = DefaultStageContextCompilerV1(rules, required_stages=("proof", "tactics"),
        native_cp_sat_authority=True, actor_observer_scope_required=True, proof_request=request,
        tactical_detector_profile=detector_profile)
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True, tactical_detector_profile=detector_profile)
    def evaluate(outcome, *, candidate=result.candidate, supplied_proof=proof):
        check = _scenario(origin={"kind": candidate.origin, **(
            {"exactTracePrefix": prefix.to_dict()} if candidate.origin == "exact_trace_prefix" else {})},
            constraints={**root_claims[0], "predicateId": "PRED-ROOT-CHECK-COUNT"}, outcome=outcome)
        bindings = provider(candidate, supplied_proof, information_graph=graph)
        return compiler.evaluate(check, candidate, bindings, maximum_seconds=10)
    for claim in root_claims:
        assert evaluate({"op": "not", "children": [claim]}).status == "target_miss"
    causal_claim = copy.deepcopy(root_claims[-1])
    causal_claim['arguments']['causalRequired'] = True
    assert evaluate(causal_claim).status == 'target_miss'
    wrong_context = copy.deepcopy(root_claims[-1])
    wrong_context["arguments"]["actingSeat"] = "west"
    assert evaluate(wrong_context).status == "target_miss"
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"})
    search_candidate = executor_module._rebind_search_candidate_to_decision(
        replace(result.candidate, origin="search"), prefix, normalized,
        preserve_construction_outcome=True, native_cp_sat=True)
    for claim in (root_claims[1], root_claims[2], root_claims[-1]):
        assert evaluate(claim, candidate=search_candidate).status == "satisfied"
    # A trace's fallback contract cannot replace the proof-owned strength table.
    changed_trace = {**result.candidate.trace, "auction": {
        "completion": {"contract": "hokum", "trumpSuit": "spades"}}}
    assert evaluate(root_claims[-1], candidate=replace(result.candidate, trace=changed_trace)).status == "satisfied"
    for hardness in ("hard", "report", "prefer"):
        check = _scenario(constraints=root_claims[0], outcome={**root_claims[-1], "hardness": hardness})
        for mutation in ("continuation", "strengths", "equation", "card", "actor", "original_owner", "ekka", "present",
                "root_actor_is_target_variable", "target_seat_variable", "native_cp_sat_authority", "tactical_detector_profile"):
            bindings = provider(result.candidate, proof, information_graph=graph)
            context = bindings["proof"].context
            if mutation == "continuation":
                context.principal_variation_has_continuation_variable = context.model.new_constant(1)
            elif mutation == "strengths":
                context.card_strength_by_id = tuple(reversed(context.card_strength_by_id))
            elif mutation in {"card", "actor", "original_owner", "ekka", "present"}:
                first, second = context.root_actions
                context.root_actions = (replace(first, **{mutation: getattr(second, mutation)}), second)
            elif mutation in {"root_actor_is_target_variable", "target_seat_variable"}:
                setattr(context, mutation, context.model.new_constant(1))
            elif mutation == "native_cp_sat_authority":
                context.native_cp_sat_authority = False
            elif mutation == "tactical_detector_profile":
                context.tactical_detector_profile = replace(detector_profile, low_card_max_strength=36)
            else:
                remaining_index = next(index for index, variable in enumerate(context.model.proto.variables)
                    if variable.name == "qualification_remaining_plays")
                equation = next(constraint for constraint in context.model.proto.constraints
                    if constraint.has_linear() and remaining_index in constraint.linear.vars)
                equation.linear.domain[0] += 1
            assert compiler.evaluate(check, result.candidate, bindings, maximum_seconds=10).status == "incomplete"
    bindings = provider(result.candidate, proof, information_graph=graph)
    bindings["proof"].context.model.proto.variables[0].name = "tampered_root"
    tampered_scenario = _scenario(constraints=root_claims[0], outcome=root_claims[1])
    tampered = compiler.evaluate(tampered_scenario, result.candidate, bindings, maximum_seconds=10)
    assert tampered.status == "incomplete" and tampered.named_core == ("PREDICATE-PRED-ROOT-COUNT",)
    for candidate, supplied_graph, supplied_proof in (
        (replace(result.candidate, exact_state_id="STATE-OTHER"), graph, proof),
        (result.candidate, replace(graph, hidden_worlds={**graph.hidden_worlds, "complete": False}), proof),
        (result.candidate, graph, replace(proof, completion_status="proof_incomplete")),
        (result.candidate, graph, replace(proof, model_hash="0" * 64)),
    ):
        with pytest.raises(ValueError):
            provider(candidate, supplied_proof, information_graph=supplied_graph)
    incomplete = executor.execute(scenario, request, **{**limits, "maximum_hidden_worlds": 1})
    assert incomplete.status == "incomplete" and incomplete.proof is None
    failed_request = request.to_dict()
    failed_request.pop("sha256")
    failed_request.update(claim="outcome_reachable", successThreshold=1000)
    failed_scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-ZERO-ROOTS", "proof.winning_root_action_count", {"minimum": 0, "maximum": 0}),
        outcome=_leaf("PRED-NOT-APPLICABLE", "tactic.applicable", {
            "kind": "only_winning_move", "applicable": False}))
    failed = executor.execute(failed_scenario, ScenarioProofRequestV1.build(failed_request), **limits)
    assert failed.status == "proof_failed", failed.execution_receipt
    assert failed.predicate_evaluation.status == "satisfied"

    contract_request = request.to_dict()
    contract_request.pop("sha256")
    contract_request.update(utility="contract_success", utilityEncoding="signed_win_loss_v1", successThreshold=1)
    contract_request = ScenarioProofRequestV1.build(contract_request)
    debate = _leaf("PRED-ROOT-DEBATE", "tactic.verified", {
        "kind": "debate", "minimumScore": 100, "minimumEvidence": 1, "causalRequired": False})
    rescue = _leaf("PRED-ROOT-RESCUE", "tactic.verified", {
        "kind": "contract_rescue", "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True})
    contract_scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-ROOT-CONTRACT-PHASE", "state.phase", {"state": "playing"}),
        outcome={"op": "all", "children": [debate,
            _leaf("PRED-ROOT-DEBATE-APPLICABLE", "tactic.applicable", {"kind": "debate", "applicable": True}),
            {**root_claims[-1], "hardness": "report"},
            {**rescue, "hardness": "report"}]})
    contract_result = executor.execute(contract_scenario, contract_request, **limits)
    assert contract_result.status == "qualified", contract_result.execution_receipt
    assert contract_result.proof["hiddenWorlds"]["count"] == 2
    assert {row["action"]: row["value"] for row in contract_result.proof["rootActions"]} == {"S7": 1, "SA": 1}
    contract_evidence = {row["predicateId"]: row for row in contract_result.predicate_evaluation.evidence}
    assert contract_evidence["PRED-ROOT-DEBATE"]["satisfied"] is True
    assert contract_evidence["PRED-ROOT-DEBATE"]["observed"]["verified"] is True
    assert contract_evidence["PRED-ROOT-RESCUE"]["satisfied"] is False
    assert contract_evidence["PRED-ROOT-SMALL-VERIFIED"]["observed"]["applicable"] is True
    assert contract_evidence["PRED-ROOT-SMALL-VERIFIED"]["observed"]["verified"] is False
    contract_bindings = DefaultStageContextCompilerV1(rules, required_stages=("proof", "tactics"),
        native_cp_sat_authority=True, actor_observer_scope_required=True, proof_request=contract_request)(
            contract_result.candidate, captured["proof"], information_graph=captured["graph"])
    assert contract_bindings["proof"].context.proof_utility == "contract_success"
    rescue_scenario = _scenario(constraints=_leaf("PRED-ROOT-CONTRACT-COUNT", "proof.winning_root_action_count",
        {"minimum": 2, "maximum": 2}, "report"), outcome=rescue)
    assert compiler.evaluate(rescue_scenario, contract_result.candidate, contract_bindings,
        maximum_seconds=10).status == "target_miss"

    kabout_request = request.to_dict()
    kabout_request.pop("sha256")
    kabout_request.update(utility="kabout", successThreshold=1)
    kabout_request = ScenarioProofRequestV1.build(kabout_request)
    kabout_scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-KABOUT-PHASE", "state.phase", {"state": "playing"}),
        outcome={"op": "all", "children": [claim for kind in ("kabout_creation", "kabout_preservation", "kabout_prevention")
            for claim in (
                _leaf(f"PRED-{kind.upper().replace('_', '-')}-APPLICABLE", "tactic.applicable", {"kind": kind, "applicable": False}),
                _leaf(f"PRED-{kind.upper().replace('_', '-')}-VERIFIED", "tactic.verified", {"kind": kind,
                    "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True}, "report"))]})
    kabout_result = executor.execute(kabout_scenario, kabout_request, **limits)
    assert kabout_result.status == "qualified", kabout_result.execution_receipt
    assert kabout_result.proof["hiddenWorlds"]["count"] == 2
    completed = kabout_result.candidate.state["completedTricks"]
    expected = [sum(int(Seat.parse(row["winner"]).team) == team for row in completed) for team in range(2)]
    assert all(expected)
    for row in kabout_result.predicate_evaluation.evidence:
        if row["kind"] not in {"tactic.applicable", "tactic.verified"}:
            continue
        observed = row["observed"]
        assert observed["applicable"] is False and observed["verified"] is False
        assert [observed["baseNorthSouthTricks"], observed["baseWestEastTricks"]] == expected
        assert observed["alternativeGuaranteeScope"] == (
            "uniform_target_kabout_guarantee" if observed["detectorId"] == "kabout_prevention"
            else "no_uniform_target_kabout_guarantee")
    kabout_provider = DefaultStageContextCompilerV1(rules, required_stages=("proof", "tactics"),
        native_cp_sat_authority=True, actor_observer_scope_required=True, proof_request=kabout_request)
    with pytest.raises(ValueError, match="ownership"):
        kabout_provider(replace(kabout_result.candidate, state={**kabout_result.candidate.state,
            "completedTricks": []}), captured["proof"], information_graph=captured["graph"])


def test_native_actor_final_play_has_no_structural_continuation(root, monkeypatch):
    from ortools.sat.python import cp_model
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = json.loads((root / "tests/fixtures/research-native-qualified-round-v3.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    prefix["playActions"] = prefix["playActions"][:31]
    prefix["stateAssertions"] = {}
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        constraints=_leaf("PRED-FINAL-PHASE", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-FINAL-COUNT", "proof.winning_root_action_count", {"minimum": 0, "maximum": 1}))
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": "north_south", "utility": "round_win",
        "successThreshold": 1, "budgetProfile": "actor_information_set_exact_v1"})
    captured = {}
    original_compile, original_prove = executor_module.compile_observed_decision_v1, executor_module.prove_actor_information_set_v1
    def compile_graph(*args, **kwargs):
        result = original_compile(*args, **kwargs)
        captured["graph"] = result.graph
        return result
    def prove(*args, **kwargs):
        result = original_prove(*args, **kwargs)
        captured["proof"] = result
        return result
    monkeypatch.setattr(executor_module, "compile_observed_decision_v1", compile_graph)
    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", prove)
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-ACTOR-FINAL", bundle_hash="a" * 64)
    result = executor.execute(scenario, request, maximum_proof_nodes=1000, maximum_proof_depth=1,
        maximum_hidden_worlds=2, maximum_seconds=15)
    assert result.status == "qualified", result.execution_receipt
    context = DefaultStageContextCompilerV1(rules, required_stages=("proof",), native_cp_sat_authority=True,
        actor_observer_scope_required=True, proof_request=request)(
            result.candidate, captured["proof"], information_graph=captured["graph"])["proof"].context
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert solver.value(context.principal_variation_has_continuation_variable) == 0


def test_native_actor_kabout_prevention_has_a_real_uniform_stopping_certificate(root, monkeypatch):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    case = next(row for row in json.loads((root / "tests/fixtures/audit5_public_baseline.json").read_text())
        if row["seed"] == 731186)
    plays = [row for trick in case["history"] for row in trick] + case["current"]
    hands = {seat: [row["card"] for row in plays if row["actor"] == seat.name.lower()] for seat in Seat}
    for seat, card in ((Seat.NORTH, "CQ"), (Seat.EAST, "CK"), (Seat.SOUTH, "DQ")):
        hands[seat].append(card)
    hands[Seat.WEST].extend(case["hand"])
    assert all(len(cards) == 8 for cards in hands.values())
    assert len({card for cards in hands.values() for card in cards}) == 32
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.SOUTH, buyer=Seat.EAST)
    deck = [None] * 32
    for seat, cards in hands.items():
        for position, card in zip(layout.positions_for(seat, final=True), cards, strict=True):
            deck[position] = Card.parse(card).id
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": deck, "dealer": "south",
        "auctionActions": [{"actor": "east", "action": "sun"}], "escalationActions": [],
        "projectDeclarations": [{"owner": "north", "kind": "fifty", "value": 10, "highRank": 7, "declarationOrder": 1}],
        "playActions": [{"actor": row["actor"], "card": row["card"], "ekka": False} for row in plays]})
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "exact", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "kabout", "successThreshold": 1, "budgetProfile": "actor_information_set_exact_v1"})
    scenario = _scenario(origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        constraints=_leaf("PRED-STOP-PHASE", "state.phase", {"state": "playing"}),
        outcome={"op": "all", "children": [
            _leaf("PRED-STOP-APPLICABLE", "tactic.applicable", {"kind": "kabout_prevention", "applicable": True}),
            _leaf("PRED-STOP-VERIFIED", "tactic.verified", {"kind": "kabout_prevention", "minimumScore": 100,
                "minimumEvidence": 1, "causalRequired": True})]})
    captured = {}
    original = executor_module.prove_actor_information_set_v1
    def prove(*args, **kwargs):
        assert kwargs["native_opponent_stop"] is True
        result = original(*args, **kwargs)
        captured["proof"] = result
        return result
    monkeypatch.setattr(executor_module, "prove_actor_information_set_v1", prove)
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-ACTOR-STOP", bundle_hash="a" * 64).execute(scenario, request,
            maximum_proof_nodes=100_000, maximum_proof_depth=8, maximum_hidden_worlds=64, maximum_seconds=30)
    assert result.status == "qualified", result.execution_receipt
    assert result.proof["hiddenWorlds"]["complete"] is True and result.proof["hiddenWorlds"]["count"] == 2
    assert result.proof["selectedAction"] == "C10"
    assert {row["action"]: row["value"] for row in result.proof["rootActions"]} == {"C10": -1, "DK": 1}
    certificate = captured["proof"].model_document()["opponentStopCertificate"]
    assert {row["rootAction"]: row["robustValue"] for row in certificate["roots"]} == {"C10": 1, "DK": -1}
    observed = next(row["observed"] for row in result.predicate_evaluation.evidence
        if row["predicateId"] == "PRED-STOP-VERIFIED")
    assert observed["verified"] is True and observed["causalProven"] is True
    assert observed["qualificationSource"] == "native_cp_sat_opponent_stop_and_public_trick_variables"
    assert observed["selectedGuaranteeScope"] == "uniform_opponent_no_target_kabout_policy"
    assert observed["stoppingVariables"] == [row["winningVariable"] for row in certificate["rootActionVariables"]]

    from baloot_platform.research.ortools_information_set_proof import (
        OrToolsActorInformationSetProofV1, _FiniteProofBudgetExceeded,
    )
    def stop_companion(*args, **kwargs):
        raise _FiniteProofBudgetExceeded("cancelled_or_wall_time")
    with monkeypatch.context() as patch:
        patch.setattr(OrToolsActorInformationSetProofV1, "_opponent_stop_problem", staticmethod(stop_companion))
        incomplete = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
            bundle_revision_id="CBR-ACTOR-STOP", bundle_hash="a" * 64).execute(scenario, request,
                maximum_proof_nodes=100_000, maximum_proof_depth=8, maximum_hidden_worlds=64, maximum_seconds=30)
    assert incomplete.status == "incomplete" and incomplete.predicate_evaluation is None
    assert incomplete.proof["completionStatus"] == "exact_complete"
    assert incomplete.candidate is not None
    artifacts = incomplete.semantic_artifact_documents()
    assert canonical_sha256(artifacts["proofGraph"]) == incomplete.proof["identities"]["graphHash"]
    assert artifacts["proofModel"]["opponentStopCertificate"]["completionStatus"] == "proof_incomplete"
    assert incomplete.execution_receipt["incompleteCompilation"]["stage"] == "opponent_stopping_proof"
    assert incomplete.execution_receipt["incompleteCompilation"]["exhaustionReason"] == "cancelled_or_wall_time"
    assert captured["proof"].model_document()["opponentStopCertificate"]["completionStatus"] == "proof_incomplete"


def test_actor_model_error_enters_compiler_disagreement_quarantine(
    monkeypatch,
    root: Path,
    rules,
):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = copy.deepcopy(fixture["exactTracePrefix"])
    prefix["playActions"].extend(
        (
            {"actor": "west", "card": "CA"},
            {"actor": "east", "card": "C8"},
            {"actor": "north", "card": "C7"},
            {"actor": "west", "card": "D7"},
            {"actor": "south", "card": "S7"},
        )
    )
    prefix["stateAssertions"] = {}
    scenario = _scenario(
        origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        constraints=_leaf("PRED-INFO-ERROR", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-INFO-ERROR-NEXT", "state.next_actor", {"seat": "east"}),
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 1,
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )

    def model_error(compilation, request, **_kwargs):
        document = {
            "schemaVersion": 1,
            "kind": "actor_information_set_cegis_model_v1",
            "completionStatus": "model_error",
            "modelStatus": "quarantined",
            "quarantineReason": "observation_action_domain_mismatch",
        }
        model_hash = canonical_sha256(document)
        return OrToolsGraphProofOutcomeV1.build(
            completion_status="model_error",
            solver_status="model_error",
            claim_satisfied=False,
            selected_action=None,
            winning_root_action_count=None,
            root_actions=(),
            principal_variation=(),
            continuation_lines=None,
            policies=(),
            graph_hash=compilation.graph.to_dict()["sha256"],
            model_hash=model_hash,
            request_hash=request.sha256,
            variable_count=0,
            constraint_count=0,
            model_document=document,
            hidden_worlds=compilation.graph.hidden_worlds,
            proof_engine_id="ortools_actor_information_set_cegis_v1",
            proof_engine_version="1.0.0",
        )

    monkeypatch.setattr(
        "baloot_platform.research.ortools_research_executor.prove_actor_information_set_v1",
        model_error,
    )
    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=4,
        maximum_hidden_worlds=64,
        maximum_seconds=15,
    )

    assert result.status == "compiler_disagreement"
    assert result.proof["completionStatus"] == "model_error"
    assert result.proof["completionStatus"] != "proof_failed"
    assert result.rejection_hashes == ()


def test_actor_information_set_continuation_count_fails_preflight(root: Path, rules):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    scenario = _scenario(
        origin={
            "kind": "exact_trace_prefix",
            "exactTracePrefix": fixture["exactTracePrefix"],
        },
        constraints=_leaf("PRED-INFO-COUNT", "state.phase", {"state": "playing"}),
        outcome=_leaf("PRED-INFO-NEXT", "state.next_actor", {"seat": "west"}),
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "winning_continuation_line_count",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )
    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_hidden_worlds=64,
    )
    assert result.status == "incomplete"
    assert result.preflight.unsupported == (
        {
            "predicateId": "PROOF-REQUEST",
            "code": "actor_information_set_continuation_count_unsupported",
        },
    )


def test_default_search_lifts_decision_prefix_once_and_proves_all_continuations(
    monkeypatch,
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)
    original_normalize = OrToolsExactStateNormalizerV1.normalize
    normalization_calls = 0

    def counted_normalize(self, *args, **kwargs):
        nonlocal normalization_calls
        normalization_calls += 1
        return original_normalize(self, *args, **kwargs)

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", counted_normalize)
    result = _executor(rules).execute(
        scenario,
        _proof(target_team=complete_candidate.trace["score"]["roundWinner"]),
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_candidates=1,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert normalization_calls == 2
    assert result.execution_receipt["completedCandidates"] == 1
    assert result.candidate.exact_state_id.startswith("STATE-")
    assert result.candidate.trace["exactNormalization"]["exactStateId"] == (
        result.candidate.exact_state_id
    )
    assert (
        result.candidate.state == result.candidate.trace["exactNormalization"]["state"]
    )
    assert (
        result.candidate.state["currentPlayer"]
        == result.candidate.trace["exactNormalization"]["nextActor"]
    )
    exact_prefix = result.candidate.trace["exactTracePrefix"]
    assert len(exact_prefix["playActions"]) == 24
    assert len(result.candidate.trace["play"]["plays"]) == 32
    assert result.proof["completionStatus"] == "exact_complete"


def test_search_candidate_can_run_exact_actor_information_proof(
    rules,
    complete_candidate,
):
    actor = complete_candidate.trace["play"]["plays"][28]["actor"]
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-ACTOR-SEARCH-CONTRACT",
                "auction.contract",
                {"contract": complete_candidate.trace["contract"]},
            ),
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "trickIndex": 7,
                    "actingSeat": actor,
                    "remainingCards": 1,
                },
            },
            "outcome": _leaf(
                "PRED-ACTOR-SEARCH-OUTCOME",
                "outcome.team_round",
                {
                    "team": complete_candidate.trace["score"]["roundWinner"],
                    "status": "won",
                },
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": complete_candidate.trace["score"]["roundWinner"],
            "utility": "round_win",
            "successThreshold": 1,
            "budgetProfile": "actor_information_set_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=1_000,
        maximum_proof_depth=8,
        maximum_hidden_worlds=64,
        maximum_candidates=1,
        maximum_seconds=30,
    )

    assert result.preflight.supported is True
    assert result.status in {"qualified", "proof_failed"}, json.dumps(
        result.execution_receipt, sort_keys=True
    )
    assert result.proof["completionStatus"] == "exact_complete"
    assert result.proof["proofEngine"]["id"] == (
        "ortools_actor_information_set_cegis_v1"
    )
    assert result.proof["hiddenWorlds"]["complete"] is True
    assert result.proof["hiddenWorlds"]["count"] <= 64


def test_search_candidate_can_run_policy_conditioned_play_proof(
    rules,
    complete_candidate,
):
    scenario = _matching_scenario(complete_candidate)
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": complete_candidate.trace["score"]["roundWinner"],
            "utility": "round_win",
            "successThreshold": -1,
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_candidates=1,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.candidate.origin == "search"
    assert result.proof["completionStatus"] == "policy_complete"
    assert len(result.candidate.trace["exactTracePrefix"]["playActions"]) == 24
    assert len(result.candidate.trace["completeExactTracePrefix"]["playActions"]) == 32
    assert result.candidate.trace["completeExactNormalization"]["phase"] == "scored"
    assert (
        result.candidate.trace["policyContinuation"]["proofOutcomeHash"]
        == (result.proof["sha256"])
    )


def test_search_target_miss_statuses_and_default_proof_refusal_are_honest(
    rules,
    complete_candidate,
):
    miss = _scenario(
        constraints=_leaf(
            "PRED-SEARCH-MISS",
            "state.phase",
            {"state": "created"},
        ),
        outcome=_leaf(
            "PRED-SEARCH-MISS-REPORT",
            "outcome.team_round",
            {
                "team": complete_candidate.trace["score"]["roundWinner"],
                "status": "won",
            },
            "report",
        ),
    )

    def feasible(*_args, **_kwargs):
        return OrToolsCandidateAttemptV1("feasible", complete_candidate)

    semantic_miss = _executor(rules).execute(
        miss,
        _proof(),
        candidate_compiler=feasible,
        maximum_candidates=1,
    )
    assert semantic_miss.status == "semantic_postcondition_miss"
    assert semantic_miss.execution_receipt["completedCandidates"] == 1

    attempts = iter(
        (
            OrToolsCandidateAttemptV1("feasible", complete_candidate),
            OrToolsCandidateAttemptV1("infeasible", None, "exhausted"),
        )
    )
    not_found = _executor(rules).execute(
        miss,
        _proof(),
        candidate_compiler=lambda *_args, **_kwargs: next(attempts),
        maximum_candidates=2,
    )
    assert not_found.status == "not_found"
    assert not_found.execution_receipt["completedCandidates"] == 1

    actor_information = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "strategyClass": "pure_deterministic",
            "budgetProfile": "test_exact",
        }
    )
    refused = _executor(rules).execute(
        _matching_scenario(complete_candidate),
        actor_information,
        maximum_proof_nodes=10,
        maximum_proof_depth=1,
    )
    assert refused.status == "incomplete"
    assert refused.candidate is None
    assert {row["code"] for row in refused.preflight.unsupported} == {
        "actor_information_set_requires_world_budget"
    }


@pytest.mark.parametrize(
    ("selected", "universe", "code"),
    [
        (None, "all_legal", "continuation_count_requires_selected_action"),
        ("HA", "optimal_strategy", "continuation_count_line_universe_mismatch"),
    ],
)
def test_native_hard_line_count_rejects_unsupported_request_before_search(
    root, selected, universe, code
) -> None:
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _scenario(
        constraints=_leaf(
            "PRED-LINE-COUNT", "proof.winning_continuation_line_count",
            {"lineUniverse": "all_legal", "minimum": "0", "maximum": "2"},
        ),
        outcome=_leaf(
            "PRED-OUTCOME", "outcome.team_round",
            {"team": "north_south", "status": "won"}, "report",
        ),
    )
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1,
        "mode": "exact",
        "visibility": "complete_state",
        "claim": "winning_continuation_line_count",
        "targetTeam": "north_south",
        "utility": "round_win",
        "successThreshold": 1,
        "lineUniverse": universe,
        **({"selectedAction": selected} if selected is not None else {}),
        "budgetProfile": "late_game_exact_v1",
    })
    result = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-LINE-COUNT-PREFLIGHT", bundle_hash="a" * 64,
    ).execute(scenario, request, maximum_seconds=1)
    assert result.status == "incomplete"
    assert result.candidate is None
    assert {item["code"] for item in result.preflight.unsupported} == {code}


@pytest.mark.parametrize("native", [False, True])
def test_exact_bid_policy_proof_qualifies_decision_and_selected_completion(rules, native):
    if native:
        rules = RuleBook.load(Path("configs/rules/saudi_federation_2023_v2.yaml"))
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix.update(
        {
            "auctionActions": [
                {"actor": actor, "action": "pass"}
                for actor in (
                    "north",
                    "west",
                    "south",
                    "east",
                    "north",
                    "west",
                    "south",
                )
            ],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _leaf(
                "PRED-BID-DECISION-PHASE",
                "state.phase",
                {"state": "bidding"},
            ),
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 7}},
            "outcome": _leaf(
                "PRED-BID-COMPLETION-MARGIN",
                "outcome.round_margin",
                {
                    "team": "north_south",
                    "minimum": -100000,
                    "maximum": 100000,
                },
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "selectedAction": "sun",
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    if native:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [raw["constraints"],
            _leaf("PRED-POLICY-LINE", "proof.winning_continuation_line_count",
                  {"minimum": "1", "maximum": "1", "lineUniverse": "policy_conditioned"})]}
        scenario = ResearchScenarioV1.build(raw)
    executor = (OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-TEST-NATIVE-POLICY", bundle_hash="a" * 64) if native else _executor(rules))
    result = executor.execute(
        scenario,
        proof,
        maximum_proof_nodes=100,
        maximum_proof_depth=16,
        maximum_seconds=90,
    )

    assert result.status == "qualified"
    assert result.proof["completionStatus"] == "policy_complete"
    assert len(result.proof["rootActions"]) == 5
    assert result.proof["continuationLines"]["exactCount"] == "1"
    assert result.candidate.state["phase"] == "bidding"
    assert result.candidate.trace["exactNormalization"]["nextActor"] == "east"
    continuation = result.candidate.trace["policyContinuation"]
    assert continuation["rootAction"] == "sun"
    assert continuation.get("allPassed") is not True
    assert len(continuation["exactTracePrefix"]["playActions"]) == 32
    assert result.candidate.trace["completeExactNormalization"]["phase"] == "scored"
    assert "auctionCompletion" in result.candidate.stage_receipts
    assert "outcome" in result.candidate.stage_receipts


@pytest.mark.parametrize("decision_kind", ["bid", "escalation"])
def test_native_preplay_policy_proof_qualifies_kabout_predicates(root, decision_kind):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))
    prefix = fixture["exactTracePrefix"]
    prefix.update(matchScore=[101, 0], escalationActions=[], projectActions=[],
        projectDeclarations=[], projectAssertions=[], playActions=[], stateAssertions={})
    if decision_kind == "bid":
        prefix["auctionActions"] = [{"action": "pass"}] * 7
    scenario = ResearchScenarioV1.build({"schemaVersion": 1,
        "origin": {"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
        "constraints": _leaf("PRED-NO-KABOUT-YET", "tactic.applicable",
            {"kind": "kabout_preservation", "applicable": False}),
        "decisionPoint": {"kind": decision_kind, "selector": {"actionIndex": 7}
            if decision_kind == "bid" else {"actingSeat": "west"}},
        "outcome": {"op": "all", "children": [
            _leaf(f"PRED-{tactic.upper().replace('_', '-')}", "tactic.verified",
                {"kind": tactic, "minimumScore": 100, "minimumEvidence": 1,
                    "causalRequired": True}, "report")
            for tactic in ("kabout_creation", "kabout_preservation", "kabout_prevention")]}})
    proof = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "policy_conditioned",
        "visibility": "complete_state", "strategyClass": "frozen_policy",
        "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "kabout", "successThreshold": 1,
        "policies": _policy_templates(), "budgetProfile": "late_game_exact_v1"})
    result = OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-NATIVE-PREPLAY-KABOUT", bundle_hash="a" * 64).execute(
            scenario, proof, maximum_proof_nodes=50_000, maximum_proof_depth=64, maximum_seconds=90)
    assert result.status == "qualified", result.execution_receipt
    assert result.proof["completionStatus"] == "policy_complete"
    assert result.candidate.state["completedTricks"] == []
    assert result.candidate.state["currentTrick"] is None
    assert result.candidate.trace["completeExactNormalization"]["phase"] == "scored"
    evidence = result.proof["nativeQualification"]
    assert len(evidence) == 4
    assert evidence["PRED-NO-KABOUT-YET"]["satisfied"] is True
    assert evidence["PRED-NO-KABOUT-YET"]["observed"]["applicable"] is False
    for row in evidence.values():
        assert row["observed"]["selectedRootAction"] == result.proof["selectedAction"]
        assert row["observed"]["qualificationSource"].startswith("native_cp_sat_")
    reported = {row["predicateId"]: row for row in result.predicate_evaluation.evidence}
    for predicate_id in evidence:
        assert reported[predicate_id]["observed"] == evidence[predicate_id]


def test_search_bid_policy_proof_rebinds_prefix_and_selected_completion(
    rules,
    complete_candidate: OrToolsNeutralCandidateV1,
):
    complete_actions = complete_candidate.trace["auction"]["completion"]["actions"]
    action_index = len(complete_actions) - 1
    selected_row = complete_actions[action_index]
    selected_action = str(selected_row["action"])
    if selected_action == "hokum":
        selected_action += f":{selected_row['trumpSuit']}"
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-SEARCH-BID-PHASE",
                "state.phase",
                {"state": "bidding"},
            ),
            "decisionPoint": {
                "kind": "bid",
                "selector": {"actionIndex": action_index},
            },
            "outcome": _leaf(
                "PRED-SEARCH-BID-OUTCOME",
                "outcome.round_margin",
                {
                    "team": "north_south",
                    "minimum": -100000,
                    "maximum": 100000,
                },
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "selectedAction": selected_action,
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=500,
        maximum_proof_depth=16,
        maximum_seconds=90,
    )

    assert result.status == "qualified"
    assert result.candidate.origin == "search"
    exact = result.candidate.trace["exactNormalization"]
    assert exact["phase"] == "bidding"
    assert len(result.candidate.trace["exactTracePrefix"]["auctionActions"]) == (
        action_index
    )
    assert result.candidate.state == exact["state"]
    assert result.candidate.trace["policyContinuation"]["rootAction"] == (
        selected_action
    )
    assert result.candidate.trace["completeExactNormalization"]["phase"] == "scored"


def test_search_bid_bounded_proof_rebinds_prefix(
    rules,
    complete_candidate: OrToolsNeutralCandidateV1,
):
    action_index = len(complete_candidate.trace["auction"]["completion"]["actions"]) - 1
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-SEARCH-BID-BOUNDED-PHASE",
                "state.phase",
                {"state": "bidding"},
            ),
            "decisionPoint": {
                "kind": "bid",
                "selector": {"actionIndex": action_index},
            },
            "outcome": _leaf(
                "PRED-SEARCH-BID-BOUNDED-REPORT",
                "state.phase",
                {"state": "bidding"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 0,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=100,
        maximum_proof_depth=4,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.candidate.origin == "search"
    assert result.candidate.trace["exactNormalization"]["phase"] == "bidding"
    assert result.proof["completionStatus"] == "bounded_complete"
    assert result.semantic_artifact_documents()["proofGraph"]["scope"] == "bounded"


def test_exact_bid_selector_ordinal_must_match_prefix_length(rules):
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix.update(
        {
            "auctionActions": [],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _leaf(
                "PRED-BID-ORDINAL",
                "state.phase",
                {"state": "bidding"},
            ),
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 1}},
            "outcome": _leaf(
                "PRED-BID-ORDINAL-REPORT",
                "state.phase",
                {"state": "bidding"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    with pytest.raises(ValueError, match="must equal the auction prefix length"):
        _executor(rules).execute(
            scenario,
            proof,
            maximum_proof_nodes=100,
            maximum_proof_depth=16,
            maximum_seconds=30,
        )


@pytest.mark.parametrize("native,false_count,line_count,origin_kind", [
    (False, False, False, "exact_trace_prefix"),
    (True, False, False, "exact_trace_prefix"),
    (True, True, False, "exact_trace_prefix"),
    (True, False, True, "exact_trace_prefix"),
    (True, True, False, "search"),
    (True, False, False, "search"),
])
def test_bid_bounded_proof_uses_the_declared_frontier(rules, native, false_count, line_count, origin_kind):
    if native:
        rules = RuleBook.load(Path("configs/rules/saudi_federation_2023_v2.yaml"))
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix.update(
        {
            "auctionActions": [],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": ({"kind": "exact_trace_prefix", "exactTracePrefix": prefix}
                       if origin_kind == "exact_trace_prefix" else {"kind": "search"}),
            "constraints": _leaf(
                "PRED-BID-BOUNDED-PHASE",
                "state.phase",
                {"state": "bidding"},
            ),
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 0}},
            "outcome": _leaf(
                "PRED-BID-BOUNDED-REPORT",
                "state.phase",
                {"state": "bidding"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 0,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
    )

    if native:
        raw = scenario.to_dict()
        raw.pop("sha256")
        count_kind = "proof.winning_continuation_line_count" if line_count else "proof.winning_root_action_count"
        count_arguments = ({"minimum": "1", "maximum": "1", "lineUniverse": "all_legal"}
            if line_count else {"minimum": 0, "maximum": 0} if false_count else {"minimum": 1, "maximum": 32})
        raw["constraints"] = {"op": "all", "children": [raw["constraints"],
            _leaf("PRED-BID-COUNT", count_kind, count_arguments)]}
        scenario = ResearchScenarioV1.build(raw)
    if line_count:
        raw_proof = proof.to_dict()
        raw_proof.pop("sha256")
        raw_proof.update(selectedAction="pass", lineUniverse="all_legal")
        proof = ScenarioProofRequestV1.build(raw_proof)
    executor = (OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-TEST-NATIVE-ROOT", bundle_hash="a" * 64) if native else _executor(rules))
    if origin_kind == "search" and not false_count:
        campaign = executor.execute_campaign(
            scenario,
            proof,
            result_count=2,
            maximum_candidates=2,
            symmetry_policy_id="rules_preserving_v1",
            allow_symmetry_only_variants=False,
            minimum_decision_state_distance=1,
            minimum_principal_variation_action_distance=2,
            maximum_proof_nodes=100,
            maximum_proof_depth=4,
            maximum_seconds=30,
        )
        # One-ply variations cannot satisfy a two-action diversity distance.
        assert campaign.status == "not_found_within_budget"
        assert len(campaign.accepted_results) == 1
        result = campaign.accepted_results[0]
        assert len(result.proof["principalVariation"]) == 1
    else:
        result = executor.execute(
            scenario,
            proof,
            maximum_proof_nodes=100,
            maximum_proof_depth=4,
            maximum_seconds=30,
        )

    assert result.status == ("semantic_postcondition_miss" if false_count else "qualified")
    assert result.proof["completionStatus"] == "bounded_complete"
    assert result.proof["rootActions"]
    assert result.proof["winningRootActionCount"] == len(result.proof["rootActions"])
    if line_count:
        assert result.proof["continuationLines"]["exactCount"] == "1"
    assert result.semantic_artifact_documents()["proofGraph"]["scope"] == "bounded"


@pytest.mark.parametrize(
    ("decision_kind", "selector", "predicate_kind", "arguments"),
    (
        (
            "escalation",
            {"actingSeat": "west"},
            "escalation.level",
            {"level": "normal"},
        ),
    ),
)
def test_exact_nonplay_policy_proof_reaches_terminal_score(
    rules,
    decision_kind: str,
    selector: dict,
    predicate_kind: str,
    arguments: dict,
):
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix.update(
        {
            "matchScore": [101, 0],
            "playActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "stateAssertions": {},
        }
    )
    leaf_a = _leaf("PRED-NONPLAY-EXACT-A", predicate_kind, arguments)
    leaf_b = _leaf("PRED-NONPLAY-EXACT-B", predicate_kind, arguments)
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": leaf_a,
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": leaf_b,
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=50_000,
        maximum_proof_depth=64,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.predicate_evaluation is not None
    assert result.predicate_evaluation.status == "satisfied"
    assert result.proof is not None
    assert result.proof["completionStatus"] == "policy_complete"
    assert result.proof["rootActions"]
    assert result.proof["winningRootActionCount"] == len(result.proof["rootActions"])
    assert result.candidate is not None
    assert result.candidate.trace["exactNormalization"]["legalActions"]
    continuation = result.candidate.trace["policyContinuation"]
    assert len(continuation["exactTracePrefix"]["playActions"]) == 32
    assert all(
        identity["materializedHash"]
        for identity in continuation["policyIdentities"].values()
    )
    assert (
        result.candidate.stage_receipts["policyContinuation"].evidence_hash
        == (continuation["sha256"])
    )


@pytest.mark.parametrize(
    ("native", "decision_kind", "selector", "predicate_kind", "arguments"),
    (
        (
            False,
            "escalation",
            {"actingSeat": "west"},
            "escalation.level",
            {"level": "normal"},
        ),
        (True, "escalation", {"actingSeat": "west"}, "escalation.level", {"level": "normal"}),
        (True, "bid", {"actionIndex": 7}, "state.legal_action_count", {"minimum": 1, "maximum": 8}),
    ),
)
def test_exact_input_nonplay_bounded_proof_grafts_exhaustive_play_graph(
    rules,
    native,
    decision_kind: str,
    selector: dict,
    predicate_kind: str,
    arguments: dict,
):
    if native:
        rules = RuleBook.load(Path("configs/rules/saudi_federation_2023_v2.yaml"))
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix.update(
        {
            "matchScore": [101, 0],
            "playActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "stateAssertions": {},
        }
    )
    if decision_kind == "bid":
        prefix["auctionActions"] = [{"action": "pass"} for _ in range(7)]
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _leaf("PRED-NONPLAY-BOUNDED-A", predicate_kind, arguments),
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": _leaf(
                "PRED-NONPLAY-BOUNDED-B",
                predicate_kind,
                arguments,
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "horizon": {"unit": "plies", "value": 2},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
    )

    if native:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [raw["constraints"],
            _leaf("PRED-NONPLAY-ROOT-COUNT", "proof.winning_root_action_count",
                  {"minimum": 1, "maximum": 32}),
            _leaf("PRED-NONPLAY-ROOT-TACTIC", "tactic.applicable",
                  {"kind": "only_winning_move", "applicable": True}),
            _leaf("PRED-NONPLAY-ROOT-REPORT", "tactic.verified",
                  {"kind": "only_winning_move", "minimumScore": 100, "minimumEvidence": 1}, "report")]}
        scenario = ResearchScenarioV1.build(raw)
    executor = (OrToolsUnifiedResearchExecutorV1(rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-TEST-NATIVE-NONPLAY", bundle_hash="a" * 64) if native else _executor(rules))
    result = executor.execute(
        scenario,
        proof,
        maximum_proof_nodes=1_000,
        maximum_proof_depth=4,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.proof["completionStatus"] == "bounded_complete"
    assert result.proof["principalVariation"]
    assert result.proof["rootActions"]
    if native:
        assert result.proof["nativeQualification"]["PRED-NONPLAY-ROOT-TACTIC"]["satisfied"]
        assert result.proof["nativeQualification"]["PRED-NONPLAY-ROOT-TACTIC"]["observed"]["selectedRootAction"] == result.proof["selectedAction"]
        assert result.proof["nativeQualification"]["PRED-NONPLAY-ROOT-REPORT"]["satisfied"] is False
        report = next(item for item in result.predicate_evaluation.evidence
                      if item["predicateId"] == "PRED-NONPLAY-ROOT-REPORT")
        assert report["satisfied"] is False
        assert report["evidenceExtractor"] == "cp_sat.native_proof_same_solve_v1"
        assert report["observed"] == result.proof["nativeQualification"]["PRED-NONPLAY-ROOT-REPORT"]
    graph = result.semantic_artifact_documents()["proofGraph"]
    assert graph["scope"] == "bounded"
    assert {
        node["leafKind"] for node in graph["nodes"] if node.get("utility") is not None
    } == ({"boundary", "terminal"} if decision_kind == "bid" else {"boundary"})


def test_exact_input_nonplay_global_proof_preserves_budget_incomplete(rules):
    fixture = json.loads(
        Path("examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = fixture["exactTracePrefix"]
    prefix.update(
        {
            "playActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "stateAssertions": {},
        }
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": prefix,
            },
            "constraints": _leaf(
                "PRED-NONPLAY-EXACT-BUDGET",
                "escalation.level",
                {"level": "normal"},
            ),
            "decisionPoint": {
                "kind": "escalation",
                "selector": {"actingSeat": "west"},
            },
            "outcome": _leaf(
                "PRED-NONPLAY-EXACT-BUDGET-REPORT",
                "escalation.level",
                {"level": "normal"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        maximum_proof_nodes=50,
        maximum_proof_depth=4,
        maximum_seconds=5,
    )

    assert result.status == "incomplete"
    assert result.candidate is not None
    assert result.proof is None


@pytest.mark.parametrize(
    ("decision_kind", "selector", "predicate_kind", "arguments"),
    (
        (
            "escalation",
            {"actingSeat": "west", "actionIndex": 0},
            "state.legal_action_count",
            {"minimum": 1, "maximum": 8},
        ),
    ),
)
def test_search_nonplay_policy_proof_rebinds_decision_and_terminal_outcome(
    rules,
    complete_candidate: OrToolsNeutralCandidateV1,
    decision_kind: str,
    selector: dict,
    predicate_kind: str,
    arguments: dict,
):
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-NONPLAY-SEARCH-STATE", predicate_kind, arguments
            ),
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": _leaf(
                "PRED-NONPLAY-SEARCH-OUTCOME",
                "outcome.round_margin",
                {"team": "north_south", "minimum": -100000, "maximum": 100000},
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=50_000,
        maximum_proof_depth=64,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.candidate is not None
    assert result.candidate.origin == "search"
    normalized = result.candidate.trace["exactNormalization"]
    expected_actor = selector.get("actingSeat") or selector["owner"]
    assert normalized["decisionKind"] == decision_kind
    assert normalized["nextActor"] == expected_actor
    continuation = result.candidate.trace["policyContinuation"]
    assert len(continuation["exactTracePrefix"]["playActions"]) == 32
    score = result.candidate.trace["score"]
    assert score["gamePoints"] == score["awardedPoints"]
    assert len(score["cardGamePoints"]) == 2
    assert result.candidate.stage_receipts["outcome"].evidence_hash == (
        canonical_sha256(score)
    )


@pytest.mark.parametrize(
    (
        "decision_kind",
        "selector",
        "predicate_kind",
        "arguments",
        "expected_status",
    ),
    (
        (
            "escalation",
            {"actingSeat": "west", "actionIndex": 0},
            "state.legal_action_count",
            {"minimum": 1, "maximum": 8},
            "qualified",
        ),
    ),
)
def test_search_nonplay_bounded_proof_rebinds_decision(
    rules,
    complete_candidate: OrToolsNeutralCandidateV1,
    decision_kind: str,
    selector: dict,
    predicate_kind: str,
    arguments: dict,
    expected_status: str,
):
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-NONPLAY-SEARCH-BOUNDED", predicate_kind, arguments
            ),
            "decisionPoint": {"kind": decision_kind, "selector": selector},
            "outcome": _leaf(
                "PRED-NONPLAY-SEARCH-BOUNDED-REPORT",
                "outcome.round_margin",
                {"team": "north_south", "minimum": -100000, "maximum": 100000},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "horizon": {"unit": "plies", "value": 2},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=1_000,
        maximum_proof_depth=4,
        maximum_candidates=1,
        maximum_seconds=30,
    )

    assert result.status == expected_status, json.dumps(
        result.to_dict(), sort_keys=True
    )
    assert result.candidate.origin == "search"
    if expected_status == "incomplete":
        assert (
            result.execution_receipt["incompleteCompilation"]["exhaustionReason"]
            == "graph_compilation_failed"
        )
        return
    assert result.proof["completionStatus"] == "bounded_complete"
    assert result.candidate.trace["exactNormalization"]["decisionKind"] == (
        decision_kind
    )
    assert result.semantic_artifact_documents()["proofGraph"]["scope"] == "bounded"


def test_search_nonplay_selector_miss_is_a_candidate_no_good_not_incomplete(
    rules,
    complete_candidate: OrToolsNeutralCandidateV1,
):
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-NONPLAY-MISS-STATE",
                "state.phase",
                {"state": "final_dealt"},
            ),
            "decisionPoint": {
                "kind": "escalation",
                "selector": {"actingSeat": "west", "actionIndex": 100},
            },
            "outcome": _leaf(
                "PRED-NONPLAY-MISS-REPORT",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
                "report",
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "policy_conditioned",
            "visibility": "complete_state",
            "strategyClass": "frozen_policy",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": 1,
            "lineUniverse": "policy_conditioned",
            "policies": _policy_templates(),
            "budgetProfile": "late_game_exact_v1",
        }
    )

    result = _executor(rules).execute(
        scenario,
        proof,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", complete_candidate
        ),
        maximum_proof_nodes=50_000,
        maximum_proof_depth=64,
        maximum_candidates=1,
        maximum_seconds=15,
    )

    assert result.status == "semantic_postcondition_miss"
    assert result.proof is None
    assert result.rejection_hashes == (complete_candidate.sha256,)


def test_forbidden_runtime_authorities_are_not_imported():
    source = Path(
        "src/baloot_platform/research/ortools_research_executor.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "core.engine",
        "BalootEngine",
        "HeuristicBot",
        "scenario_search",
        "proof_service",
        "runtime verifier",
    ):
        assert forbidden not in source


def test_stage_receipts_are_immutable_hash_bound():
    receipt = OrToolsResearchStageReceiptV1(
        "deal", "1" * 64, "2" * 64, "3" * 64, "optimal", 1, 2
    )
    first = receipt.to_dict()
    second = receipt.to_dict()
    assert first == second
    assert len(first["sha256"]) == 64
