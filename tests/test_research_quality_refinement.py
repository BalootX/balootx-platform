from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

import baloot_platform.research.ortools_research_executor as executor_module
from baloot_platform.core.cards import Card
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_flagship_research import OrToolsFlagshipResearchV1
from baloot_platform.research.ortools_research_executor import (
    OrToolsCandidateAttemptV1,
    OrToolsNeutralCandidateV1,
    OrToolsResearchStageReceiptV1,
    OrToolsUnifiedResearchExecutorV1,
)
from baloot_platform.research.ortools_tail_first_candidate import OrToolsTailFirstCandidateCompilerV1
from baloot_platform.research.scenario_contracts import (
    ScenarioPlayingQualityPolicyV1,
    ScenarioQualityExecutionBudgetV1,
    canonical_sha256,
)
from baloot_platform.research.tail_first_stage_planner import TailFirstStagePlannerV1
from test_research_ortools_unified_executor import _leaf, _proof, _scenario


@pytest.mark.real_clock
@pytest.mark.parametrize("case", ["wrong_hash", "expired_unknown", "success"])
def test_internal_quality_refinement_owns_misses_and_reuses_terminal_attempt(
    root, monkeypatch, case,
):
    """Exercise orchestration only; native lift/proof qualification is not mocked green."""
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "budgeted", "visibility": "actor_information_set",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": evaluator,
        } for kind, evaluator in profile["authoringDefaults"]["playingQuality"][
            "boundaryEvaluatorByDecision"].items()},
    })
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3", bundle_revision_id="CBR-REFINEMENT-TEST",
        bundle_hash="a" * 64, quality_policy=policy,
        quality_execution=ScenarioQualityExecutionBudgetV1.build(profile["qualityExecution"]),
        quality_proof_profiles=profile["proofBudgets"]["profiles"],
    )
    scenario = _scenario(constraints={"op": "all", "children": [
        _leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
        _leaf("PRED-FUTURE-ACES", "play.trick_card_count", {
            "trickIndex": 7, "selector": {"ranks": ["ace"]}, "minimum": 2, "maximum": 2,
        }),
    ]}, outcome=_leaf("PRED-REPORT", "state.phase", {"state": "playing"}, "report"),
        decision_point={"kind": "play", "selector": {
            "trickIndex": 5, "position": 3, "remainingCards": 3, "actingSeat": "west",
        }})
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario)
    assert plan.planner == "tail_first"
    dealer, contract, buyer, trump = plan.boundary_combinations[0]
    receipt = OrToolsResearchStageReceiptV1("deal", "a" * 64, "b" * 64, "c" * 64, "optimal")
    candidates = []
    for index in range(3):
        deck = tuple(range(index, 32)) + tuple(range(index))
        candidates.append(OrToolsNeutralCandidateV1("search", {
            "deckOrder": list(deck), "roundNumber": 1,
            "dealer": dealer, "contract": contract, "buyer": buyer, "trumpSuit": trump,
            "matchScore": [0, 0], "tailFirstStagePlan": plan.to_dict(),
            "tailFirstSearchCounters": {"prefixLiftAttempts": index + 1},
            "play": {"plays": [{"card": Card(card).code} for card in deck]},
        }, {}, deck, {"deal": receipt}))
    clock = [100.0]
    generated = []
    misses = []
    terminals = []
    construction_calls = []
    progress = []
    monkeypatch.setattr(executor_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(OrToolsFlagshipResearchV1, "certify_round_win_counterstrategy",
                        lambda *_args, **_kwargs: None)

    def generate(candidate, **kwargs):
        assert kwargs["scenario"] is scenario
        assert kwargs["graph_workers"] == 4
        assert 0 < kwargs["maximum_seconds"] <= 90
        generated.append(candidate.sha256)
        clock[0] += 1
        artifact = {"proofModel": {"kind": "native_bounded_play_quality_model_v1"},
                    "candidateHash": candidate.sha256}
        kwargs["checkpoint_callback"]({"kind": "quality_proof_checkpoint_v1",
            "artifact": artifact, "artifactHash": canonical_sha256(artifact)})
        kwargs["checkpoint_callback"]({"kind": "quality_solver_progress_v1", "status": "running"})
        partial = {"candidateHash": candidate.sha256, "reason": "hard_future_card_count_miss",
                   "witnessActionSelection": {"exhaustionReason": "no_optimal_witness_action"}}
        terminal = candidate is candidates[-1]
        status, reason = "not_found", "quality_generation_target_miss"
        if terminal:
            status, reason = ("feasible", None) if case == "success" else (
                "incomplete", "quality_generation_incomplete")
            partial = ({"kind": "quality_handoff_source_v1", "candidateHash": candidate.sha256}
                       if case == "success" else {
                           "candidateHash": candidate.sha256, "stage": "native_observed_play",
                           "solverStatus": "unknown", "exhaustionReason": "maximum_conflicts"})
            if case == "expired_unknown":
                clock[0] = 191.0
        elif case == "wrong_hash":
            partial["candidateHash"] = "0" * 64
        attempt = OrToolsCandidateAttemptV1(status, candidate if status == "feasible" else None,
            reason, stage_timings={"qualityAuction": 0.25}, partial_receipt={
                **partial, "sha256": canonical_sha256(partial)})
        terminals.append(attempt)
        return attempt

    def construct(compiler, _scenario, excluded, **kwargs):
        assert not excluded
        assert compiler.maximum_prefix_lift_attempts == 3
        assert kwargs["maximum_seconds"] == 90
        construction_calls.append(compiler)
        for candidate in candidates[:compiler.maximum_prefix_lift_attempts]:
            miss = compiler._candidate_refinement(candidate, 190.0 - clock[0])
            if miss is None:
                return OrToolsCandidateAttemptV1("feasible", candidate)
            assert all(row.get("completedCandidates", 0) == 0 for row in progress)
            compiler._reject_quality_candidate(candidate, reject_initial_input=True)
            misses.append(candidate)
        return OrToolsCandidateAttemptV1("incomplete", None, "prefix_lift_attempt_budget")

    def finish(status, _preflight, candidate, _evaluation, _proof, rejections,
               _started, _cpu_started, completed, **kwargs):
        return SimpleNamespace(status=status, candidate=candidate, rejections=rejections,
                               completed=completed, **kwargs)

    class ReachedQualification(Exception):
        pass

    def qualify(candidate, *_args, **_kwargs):
        assert candidate is candidates[-1]
        assert generated == [row.sha256 for row in candidates]
        assert len(construction_calls) == 1
        assert progress[-1]["completedCandidates"] == 1
        raise ReachedQualification

    monkeypatch.setattr(executor, "generate_quality_candidate", generate)
    monkeypatch.setattr(executor, "audit_quality_candidate",
                        lambda *_args, **_kwargs: pytest.fail("refinement must not audit a new line"))
    monkeypatch.setattr(executor, "_result", finish)
    monkeypatch.setattr(OrToolsTailFirstCandidateCompilerV1, "__call__", construct)
    monkeypatch.setattr(executor_module, "_search_decision_prefix", qualify)
    options = dict(maximum_candidates=1, maximum_seconds=90, maximum_prefix_lift_attempts=3,
                   maximum_proof_nodes=1_000, maximum_proof_depth=12,
                   terminal_worker_count=4, progress_callback=progress.append)
    if case == "success":
        with pytest.raises(ReachedQualification):
            executor.execute(scenario, _proof(), **options)
        return
    result = executor.execute(scenario, _proof(), **options)
    assert len(construction_calls) == 1
    expected_calls = 3 if case == "expired_unknown" else 1
    assert generated == [row.sha256 for row in candidates[:expected_calls]]
    expected_misses = 2 if case == "expired_unknown" else 0
    assert misses == candidates[:expected_misses]
    assert result.completed == 1
    if case == "expired_unknown":
        assert result.status == "incomplete"
        assert result.incomplete_receipt == terminals[-1].partial_receipt
        assert result.rejections == tuple(row.sha256 for row in misses)
        assert result.stage_timings["playingQuality"] == clock[0] - 100.0
        assert result.stage_timings["qualityAuction"] == 0.75
        assert result._quality_verified_artifacts == {canonical_sha256({
            "proofModel": {"kind": "native_bounded_play_quality_model_v1"},
            "candidateHash": candidates[-1].sha256,
        })}
