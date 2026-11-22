from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from baloot_platform.analysis.search import SearchTimeBudgetExceeded
from baloot_platform.challenge.constraint_deal_search import (
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research import service as service_module

from research_audit_trace_helpers import (
    _Harness,
    _SequentialGeneratorStub,
    _audit_items,
    _claim_case,
    _hash,
    _payload,
    _prepare_ortools_branch,
    _result_items,
    _search_receipt_items,
)
from research_audit_trace_helpers import service_harness as service_harness


def _completed_events(research: Any, claim: dict[str, Any]) -> list[dict[str, Any]]:
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["status"] == "completed"
    return list((shard.get("result") or {}).get("events") or ())


def _completed_all_pass_simulation(simulator: Any) -> Any:
    scripted_bids = tuple(
        {
            "actor": actor,
            "roundIndex": round_index,
            "action": "pass",
            "trumpSuit": None,
        }
        for round_index in (1, 2)
        for actor in ("north", "west", "south", "east")
    )
    return simulator.simulate_preordered_deck(
        91,
        tuple(range(32)),
        deal_generator_provenance={
            "schemaVersion": 1,
            "pluginId": "service_all_pass_fixture_v1",
            "profileHash": _hash("service-all-pass-profile"),
        },
        dealer=Seat.EAST,
        match_score=(0, 0),
        scripted_bids=scripted_bids,
    )


def test_scored_trace_is_retained_before_semantic_quarantine(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="semantic-quarantine",
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    call_order: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    original_put_receipt = research.store.put_sequential_search_receipt
    original_put_disagreement = research.store.put_disagreement
    original_finalize = research.audit_traces.finalize
    original_complete = research.store.complete_shard

    def put_receipt_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("receipt", args, copy.deepcopy(kwargs)))
        return original_put_receipt(*args, **kwargs)

    def put_disagreement_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("disagreement", args, copy.deepcopy(kwargs)))
        return original_put_disagreement(*args, **kwargs)

    def finalize_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("finalize", args, copy.deepcopy(kwargs)))
        return original_finalize(*args, **kwargs)

    def complete_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("complete", args, copy.deepcopy(kwargs)))
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(
        research.store,
        "put_sequential_search_receipt",
        put_receipt_spy,
    )
    monkeypatch.setattr(
        research.store,
        "put_disagreement",
        put_disagreement_spy,
    )
    monkeypatch.setattr(research.audit_traces, "finalize", finalize_spy)
    monkeypatch.setattr(research.store, "complete_shard", complete_spy)
    monkeypatch.setattr(
        research,
        "_semantic_mismatches",
        lambda *_a, **_k: ["runtime_transition_defect"],
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == service_harness.simulation.final_trace.sha256
    assert record["disposition"] == "quarantined", research.store.get_shard(
        claim["shardId"]
    )
    assert record["auditOnly"] is True
    assert record["publicationEligible"] is False
    assert {
        "stage_semantic_parity_verification",
        "runtime_transition_defect",
    }.issubset(record["failureCategories"])
    assert len(record["disagreementReferences"]) == 1
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert [item["disagreementId"] for item in disagreements] == [
        record["disagreementReferences"][0]["disagreementId"]
    ]
    assert _result_items(research, run["runCode"]) == []
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["result"]["quarantineCount"] == 1
    receipt = _search_receipt_items(research, run["runCode"])[0]
    assert receipt["attemptCount"] == claim["attemptCount"]
    assert receipt["fencingToken"] == claim["fencingToken"]
    assert receipt["admitted"] is True
    bundle = receipt["receiptBundle"]
    assert bundle["stopReason"] == "fatal_quarantine"
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0
    names = [item[0] for item in call_order]
    assert names.count("receipt") == 1
    assert names.count("disagreement") == 1
    assert names.count("finalize") == 1
    assert names.count("complete") == 1
    assert names.index("receipt") < names.index("disagreement")
    assert names.index("receipt") < names.index("finalize")
    assert names.index("receipt") < names.index("complete")
    receipt_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "receipt"
    )
    disagreement_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "disagreement"
    )
    complete_args, complete_kwargs = next(
        (args, kwargs)
        for name, args, kwargs in call_order
        if name == "complete"
    )
    expected_execution = {
        "run_code": run["runCode"],
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
        "seed": claim["seedStart"],
    }
    assert receipt_kwargs == expected_execution
    assert disagreement_kwargs == {
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
    }
    assert complete_args == (claim["shardId"],)
    assert complete_kwargs["worker_id"] == claim["leaseOwner"]
    assert complete_kwargs["fencing_token"] == claim["fencingToken"]
    assert record["source"] | {
        "runCode": run["runCode"],
        "shardId": claim["shardId"],
        "attemptCount": claim["attemptCount"],
        "fencingToken": claim["fencingToken"],
        "workerId": claim["leaseOwner"],
        "seed": claim["seedStart"],
    } == record["source"]


@pytest.mark.real_clock
def test_analysis_wall_clock_exhaustion_is_retained_as_incomplete_without_retry(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="analysis-wall-clock-incomplete",
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    analyzer_type = type(research.analyzers["screening"])

    def exhaust_wall_clock(*_args: Any, **_kwargs: Any) -> Any:
        raise SearchTimeBudgetExceeded(
            "exact search exceeded its wall-clock budget without canonical output"
        )

    monkeypatch.setattr(analyzer_type, "analyze_trace", exhaust_wall_clock)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["status"] == "completed"
    assert shard["attemptCount"] == claim["attemptCount"]
    assert generator.generate_calls == [()]
    assert _result_items(research, run["runCode"]) == []
    receipt = _search_receipt_items(research, run["runCode"])[0]
    assert receipt["receiptBundle"]["stopReason"] == "incomplete"
    assert receipt["receiptBundle"]["failureCodes"] == [
        "analysis_wall_clock_budget_exhausted"
    ]
    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "rejected"
    assert {
        "stage_candidate_analysis",
        "candidate_analysis_incomplete",
        "analysis_wall_clock_budget_exhausted",
    }.issubset(records[0]["failureCategories"])


def test_completed_trace_survives_construction_target_rejection(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    actual_buyer = Seat.parse(
        service_harness.simulation.final_trace.trusted_final_state["buyer"]
    )
    assumed_buyer = next(seat for seat in Seat if seat != actual_buyer)
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="construction-target",
        assumed_buyer=assumed_buyer,
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "rejected"
    assert {
        "stage_actor_safe_policy_target_verification",
        "construction_target_miss",
        "buyer_target_miss",
    }.issubset(records[0]["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "construction_target_miss"
        for item in _completed_events(research, claim)
    )
    receipt = _search_receipt_items(research, run["runCode"])[0]
    bundle = receipt["receiptBundle"]
    candidate = generator.candidates[0]
    expected_cut = OrToolsCpSatDealGeneratorV1.exact_no_good_cut_document(
        candidate,
        layout=None,
    )
    assert bundle["stopReason"] == "candidates_exhausted"
    assert len(bundle["rejectionFeedbackRows"]) == 1
    feedback = bundle["rejectionFeedbackRows"][0]
    assert feedback["noGoodCut"] == expected_cut
    assert feedback["noGoodCutHash"] == expected_cut["cutSha256"]
    continuation = bundle["terminalContinuationReceipt"]
    assert continuation["exhausted"] is True
    assert continuation["precedingNoGoodCutHashes"] == [
        expected_cut["cutSha256"]
    ]
    assert feedback["applicationReceipt"]["solveEvidenceKind"] == (
        "post_rejection_continuation_receipt"
    )
    assert feedback["applicationReceipt"]["solveEvidenceHash"] == continuation[
        "sha256"
    ]
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["result"]["optimalityCutCount"] == 0
    assert generator.generate_calls == [()]
    assert generator.probe_calls == [(candidate.candidate_hash,)]
    assert generator.legacy_generate_calls == 0


def test_actor_safe_contract_target_miss_is_rejected_not_quarantined(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    actual_contract = str(
        service_harness.simulation.final_trace.trusted_final_state["contract"]
    )
    requested_contract = next(
        item for item in ("sun", "hokum", "ashkal")
        if item != actual_contract
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="contract-target-miss",
        situation={"contracts": [requested_contract]},
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "rejected"
    assert {
        "stage_actor_safe_policy_target_verification",
        "construction_target_miss",
        "contract_target_miss",
    }.issubset(records[0]["failureCategories"])
    assert records[0]["disagreementReferences"] == []
    assert research.store.list_disagreements(run_code=run["runCode"]) == []
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None and shard["result"]["quarantineCount"] == 0
    event = next(
        item for item in _completed_events(research, claim)
        if item["type"] == "construction_target_miss"
    )
    assert event["reasons"] == ["contract_target_miss"]
    assert event["expected"]["contracts"] == [requested_contract]
    assert event["actual"]["contract"] == actual_contract


def test_compiled_replay_mismatch_wins_over_actor_safe_target_miss(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    actual_contract = str(
        service_harness.simulation.final_trace.trusted_final_state["contract"]
    )
    requested_contract = next(
        item for item in ("sun", "hokum", "ashkal")
        if item != actual_contract
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="contract-target-plus-play-parity",
        situation={"contracts": [requested_contract]},
    )
    monkeypatch.setattr(
        research,
        "_compiled_play_mismatches",
        lambda *_args, **_kwargs: ("compiler_parity_defect",),
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "quarantined"
    assert {
        "stage_semantic_parity_verification",
        "compiler_parity_defect",
    }.issubset(records[0]["failureCategories"])
    assert "construction_target_miss" not in records[0]["failureCategories"]
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "compiler_parity_defect"


def test_final_match_score_parity_mismatch_is_quarantined(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    trusted_state = service_harness.simulation.final_trace.trusted_final_state
    play_plan = research.play_compiler.verify_trace(
        trusted_state,
        rules=service_harness.simulator.rules,
    )
    tampered_state = copy.deepcopy(trusted_state)
    tampered_state["matchScore"][0] += 1
    tampered_simulation = SimpleNamespace(
        final_trace=SimpleNamespace(trusted_final_state=tampered_state)
    )
    mismatch = research._compiled_play_mismatches(
        play_plan,
        tampered_simulation,
        scoring_domain=ScoringDomainIRV1.from_rules(
            service_harness.simulator.rules.data
        ),
        expected_dealer=Seat.EAST,
        starting_match_score=(0, 0),
    )
    assert mismatch == ("compiler_parity_defect",)

    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="final-match-score-parity",
    )
    monkeypatch.setattr(
        research,
        "_compiled_play_mismatches",
        lambda *_args, **_kwargs: mismatch,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    assert records[0]["disposition"] == "quarantined"
    assert {
        "stage_semantic_parity_verification",
        "compiler_parity_defect",
    }.issubset(records[0]["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "compiler_parity_defect"


def test_completed_trace_survives_tactical_viability_rejection(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    profile = {
        "schemaVersion": 1,
        "id": "exact_tactical_viability_screen_v1",
        "revision": "1",
        "enabled": True,
        "maximumDecisions": 4,
        "requiredTacticsOnly": True,
        "rejectWhenExactAbsent": True,
    }
    payload = _payload(
        research,
        label="tactical-pre-screen",
        lane="organic_runtime",
        resources={"tacticalViabilityPreScreen": profile},
    )
    payload["tactics"][0]["mode"] = "qualification_gate"
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="tactical-pre-screen",
        payload=payload,
    )
    evidence = SimpleNamespace(
        rejected=True,
        required_tactics=("small_card_big_gain",),
        found_tactics=(),
        probe=SimpleNamespace(status="not_found"),
        sha256=_hash("tactical-pre-screen-evidence"),
        to_dict=lambda: {
            "schemaVersion": 1,
            "rejected": True,
            "requiredTactics": ["small_card_big_gain"],
            "foundTactics": [],
        },
    )
    monkeypatch.setattr(
        service_module,
        "TacticalViabilityPreScreenV1",
        lambda *_a, **_k: SimpleNamespace(evaluate=lambda *_x, **_y: evidence),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    record = _audit_items(research, run["runCode"])[0]
    assert record["disposition"] == "rejected"
    assert {
        "stage_tactical_viability_pre_screen",
        "tactical_viability_pre_screen_rejected",
        "required_tactic_absent:small_card_big_gain",
    }.issubset(record["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "tactical_viability_pre_screen" and item["rejected"]
        for item in _completed_events(research, claim)
    )


def test_completed_trace_survives_construction_pre_screen_rejection(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    profile = {
        "schemaVersion": 1,
        "id": "actor_safe_pre_screen_v1",
        "revision": "1",
        "enabled": True,
        "maximumPlayDecisions": 4,
        "maximumBidRegret": 0,
        "maximumPlayRegret": 0,
        "maximumProbabilityDrop": 0,
        "minimumConfidence": 0,
        "rejectCatastrophicCrossing": True,
    }
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="construction-pre-screen",
        resources={"constructionPreScreen": profile},
    )
    evidence = SimpleNamespace(
        rejected=True,
        rejection_reasons=("catastrophic_threshold_crossing",),
        sha256=_hash("construction-pre-screen-evidence"),
        to_dict=lambda: {
            "schemaVersion": 1,
            "rejected": True,
            "rejectionReasons": ["catastrophic_threshold_crossing"],
        },
    )
    monkeypatch.setattr(
        service_module,
        "ConstructionPreScreenV1",
        lambda *_a, **_k: SimpleNamespace(evaluate=lambda *_x, **_y: evidence),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    record = _audit_items(research, run["runCode"])[0]
    assert record["disposition"] == "rejected"
    assert {
        "stage_construction_pre_screen",
        "construction_pre_screen_rejected",
        "catastrophic_threshold_crossing",
    }.issubset(record["failureCategories"])
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "construction_pre_screen" and item["rejected"]
        for item in _completed_events(research, claim)
    )


def test_completed_all_pass_trace_is_retained_as_rejected_audit_evidence(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    all_pass = _completed_all_pass_simulation(service_harness.simulator)
    assert all_pass.publishable_laffa is False
    assert all_pass.final_trace.trusted_final_state["phase"] == (
        GamePhase.COMPLETE.value
    )
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="all-pass-complete",
        payload=_payload(
            research,
            label="all-pass-complete",
            lane="organic_runtime",
        ),
        simulation=all_pass,
    )

    def forbid_sequential_search(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("organic Research must not enter sequential OR search")

    monkeypatch.setattr(
        service_module,
        "run_sequential_candidate_search_v1",
        forbid_sequential_search,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == all_pass.final_trace.sha256
    assert record["status"] == "finalized"
    assert record["disposition"] == "rejected"
    assert record["attemptStatus"] == "completed"
    assert {
        "stage_organic_simulation",
        "organic_all_pass_complete_trace",
    }.issubset(record["failureCategories"])
    assert record["auditOnly"] is True
    assert record["publicationEligible"] is False
    assert record["replayValidation"]["status"] == "verified"
    assert record["observerSafety"]["status"] == "verified"
    assert _result_items(research, run["runCode"]) == []
    assert _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    ) == []
    assert any(
        item["type"] == "recipe_infeasible"
        and item["reason"] == "organic_runtime_all_pass"
        for item in _completed_events(research, claim)
    )


def test_constructed_completed_all_pass_trace_is_retained_as_rejected_audit_evidence(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    all_pass = _completed_all_pass_simulation(service_harness.simulator)
    assert all_pass.publishable_laffa is False
    assert all_pass.final_trace.trusted_final_state["phase"] == (
        GamePhase.COMPLETE.value
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="constructed-all-pass-complete",
        simulation=all_pass,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == all_pass.final_trace.sha256
    assert record["status"] == "finalized"
    assert record["disposition"] == "rejected"
    assert record["attemptStatus"] == "completed"
    assert {
        "stage_candidate_simulation",
        "all_pass_unscored_complete_trace",
    }.issubset(record["failureCategories"])
    assert record["auditOnly"] is True
    assert record["publicationEligible"] is False
    assert record["replayValidation"]["status"] == "verified"
    assert record["observerSafety"]["status"] == "verified"
    assert _result_items(research, run["runCode"]) == []
    assert any(
        item["type"] == "recipe_infeasible"
        and item["reason"] == "constructed_auction_all_pass"
        for item in _completed_events(research, claim)
    )


def test_incomplete_unscored_trace_is_not_admitted_or_publishable(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    evaluations: list[Any] = []
    original_build_evaluation = service_module.SequentialCandidateEvaluationV1.build

    def build_evaluation_spy(_cls: type[Any], **kwargs: Any) -> Any:
        evaluation = original_build_evaluation(**kwargs)
        evaluations.append(evaluation)
        return evaluation

    monkeypatch.setattr(
        service_module.SequentialCandidateEvaluationV1,
        "build",
        classmethod(build_evaluation_spy),
    )
    unscored_trace = SimpleNamespace(
        engine_version=service_harness.simulation.final_trace.engine_version,
        sha256=_hash("unscored-all-pass-trace"),
        trusted_final_state={"phase": GamePhase.BIDDING.value},
    )
    all_pass = SimpleNamespace(
        publishable_laffa=False,
        final_trace=unscored_trace,
    )
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="all-pass-unscored",
        payload=_payload(
            research,
            label="all-pass-unscored",
            lane="organic_runtime",
        ),
        simulation=all_pass,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    assert _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    ) == []
    assert len(evaluations) == 1
    assert evaluations[0].outcome.value == "incomplete"
    assert evaluations[0].failure_codes == ("authoritative_replay_incomplete",)
    assert not any(
        item["type"] == "recipe_infeasible"
        for item in _completed_events(research, claim)
    )
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["result"]["resultCount"] == 0
    assert shard["result"]["qualifiedCount"] == 0
    assert shard["result"]["rejectedCount"] == 0
    assert shard["result"]["quarantineCount"] == 0


def test_constructed_incomplete_trace_stops_without_no_good_cut(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    incomplete_trace = SimpleNamespace(
        engine_version=service_harness.simulation.final_trace.engine_version,
        sha256=_hash("constructed-incomplete-trace"),
        trusted_final_state={"phase": GamePhase.BIDDING.value},
    )
    incomplete = SimpleNamespace(
        publishable_laffa=True,
        final_trace=incomplete_trace,
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="constructed-incomplete",
        simulation=incomplete,
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    receipt = _search_receipt_items(research, run["runCode"])[0]
    bundle = receipt["receiptBundle"]
    assert bundle["stopReason"] == "incomplete"
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0


@pytest.mark.real_clock
def test_cancelled_candidate_evaluation_persists_unadmitted_receipt_without_effects(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="constructed-cancelled",
        resources={
            "auctionConstructionMode": "cp_sat_feasibility_line",
            "playConstructionMode": "cp_sat_feasibility_line",
        },
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    effect_calls: list[str] = []
    receipt_calls: list[dict[str, Any]] = []
    original_put_receipt = research.store.put_sequential_search_receipt
    original_put_result = research.store.put_result
    original_put_disagreement = research.store.put_disagreement
    original_finalize = research.audit_traces.finalize
    original_complete = research.store.complete_shard

    def put_receipt_spy(*args: Any, **kwargs: Any) -> Any:
        receipt_calls.append(copy.deepcopy(kwargs))
        return original_put_receipt(*args, **kwargs)

    def put_result_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("result")
        return original_put_result(*args, **kwargs)

    def put_disagreement_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("disagreement")
        return original_put_disagreement(*args, **kwargs)

    def finalize_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("finalize")
        return original_finalize(*args, **kwargs)

    def complete_spy(*args: Any, **kwargs: Any) -> Any:
        effect_calls.append("complete")
        return original_complete(*args, **kwargs)

    def cancel_during_auction(*_args: Any, **_kwargs: Any) -> Any:
        raise InterruptedError("service fixture cancellation")

    monkeypatch.setattr(
        research.store,
        "put_sequential_search_receipt",
        put_receipt_spy,
    )
    monkeypatch.setattr(research.store, "put_result", put_result_spy)
    monkeypatch.setattr(
        research.store,
        "put_disagreement",
        put_disagreement_spy,
    )
    monkeypatch.setattr(research.audit_traces, "finalize", finalize_spy)
    monkeypatch.setattr(research.store, "complete_shard", complete_spy)
    monkeypatch.setattr(research, "_compile_auction", cancel_during_auction)

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=None,
    )

    receipts = _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    )
    assert len(receipts) == 1
    persisted = receipts[0]
    assert persisted["admitted"] is False
    assert persisted["shardId"] == claim["shardId"]
    assert persisted["attemptCount"] == claim["attemptCount"]
    assert persisted["fencingToken"] == claim["fencingToken"]
    assert persisted["seed"] == claim["seedStart"]
    assert receipt_calls == [{
        "run_code": run["runCode"],
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
        "seed": claim["seedStart"],
    }]
    bundle = persisted["receiptBundle"]
    assert bundle["stopReason"] == "cancelled"
    assert bundle["failureCodes"] == ["candidate_evaluation_cancelled"]
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert bundle["evaluationReceipts"][0]["evaluation"]["outcome"] == (
        "cancelled"
    )
    assert effect_calls == []
    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    assert research.store.list_disagreements(run_code=run["runCode"]) == []
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["status"] == "running"
    assert shard.get("result") is None
    current_run = research.store.get_run(run["runCode"])
    assert current_run is not None
    assert current_run["resultCount"] == 0
    assert current_run["quarantineCount"] == 0
    assert "solverIncomplete" not in current_run
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0


class _ReplayInvalidTrace:
    def __init__(self, valid: Any) -> None:
        self._valid = valid
        self.sha256 = valid.sha256
        self.engine_version = valid.engine_version

    def to_dict(self, *, trusted: bool = False) -> dict[str, Any]:
        value = copy.deepcopy(self._valid.to_dict(trusted=True))
        value["stateHash"] = "0" * 64
        return value


def test_replay_invalid_trace_is_quarantined_without_admission_or_result(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    invalid = SimpleNamespace(
        publishable_laffa=True,
        final_trace=_ReplayInvalidTrace(service_harness.simulation.final_trace),
    )
    run, claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="replay-invalid",
        payload=_payload(
            research,
            label="replay-invalid",
            lane="organic_runtime",
        ),
        simulation=invalid,
    )

    research._execute_claimed_shard(
        run["runCode"], claim, claim["leaseOwner"], context=None
    )

    assert _audit_items(research, run["runCode"]) == []
    assert _result_items(research, run["runCode"]) == []
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    assert disagreements[0]["category"] == "runtime_transition_defect"
    assert disagreements[0]["runtimeEvidence"]["stage"] == (
        "authoritative_complete_trace_replay"
    )
    assert any(
        item["type"] == "semantic_replay_quarantine"
        and item["category"] == "runtime_transition_defect"
        for item in _completed_events(research, claim)
    )
