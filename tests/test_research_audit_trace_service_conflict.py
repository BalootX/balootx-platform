from __future__ import annotations

import copy
from typing import Any

import pytest

from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.research import service as service_module

from research_audit_trace_helpers import (
    _Harness,
    _SequenceSimulator,
    _SequentialGeneratorStub,
    _audit_items,
    _prepare_ortools_branch,
    _result_items,
    _search_receipt_items,
)
from research_audit_trace_helpers import service_harness as service_harness


def test_qualified_repeat_source_conflict_quarantines_primary_audit(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    selection_value = research.decision_score_profile.to_dict()
    selection_value.pop("sha256")
    selection_value.update({
        "id": "qualified_repeat_fixture",
        "revision": "1",
        "minimumScore": 0.0,
        "requireVerifiedTacticalPlay": False,
        "minimumSurroundingQuality": 0.0,
        "minimumSurroundingConfidence": 0.0,
        "minimumSurroundingMateriality": 0.0,
        "maximumSurroundingMaterialMistakes": 2_147_483_647,
        "maximumSurroundingUnverifiedDecisions": 2_147_483_647,
        "maximumSurroundingCatastrophicCrossings": 2_147_483_647,
    })
    decision_score_profile = SelectionProfileV1.build(selection_value)
    monkeypatch.setattr(
        research,
        "decision_score_profile",
        decision_score_profile,
    )
    monkeypatch.setattr(
        research,
        "decision_score_profile_hash",
        decision_score_profile.sha256,
    )
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="qualified-repeat-success",
        resources={"analysisMode": "screening"},
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    base_qualification = research._qualification
    base_evaluate_decision_selection = service_module.evaluate_decision_selection_v1
    qualification_calls: list[dict[str, Any]] = []
    decision_selection_calls: list[dict[str, Any]] = []

    def qualify_with_completed_content_gates(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        value = base_qualification(*args, **kwargs)
        qualification_calls.append(copy.deepcopy(value))
        return {
            **value,
            "qualified": True,
            "truthLabel": "provisional",
            "failures": [],
        }

    def record_decision_selection(
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        value = base_evaluate_decision_selection(*args, **kwargs)
        decision_selection_calls.append(copy.deepcopy(value))
        return value

    call_order: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    original_put_receipt = research.store.put_sequential_search_receipt
    original_put_result = research.store.put_result
    original_put_disagreement = research.store.put_disagreement
    original_finalize = research.audit_traces.finalize
    original_complete = research.store.complete_shard

    def put_receipt_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("receipt", args, copy.deepcopy(kwargs)))
        return original_put_receipt(*args, **kwargs)

    def put_result_spy(*args: Any, **kwargs: Any) -> Any:
        call_order.append(("result", args, copy.deepcopy(kwargs)))
        return original_put_result(*args, **kwargs)

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
        research,
        "_qualification",
        qualify_with_completed_content_gates,
    )
    monkeypatch.setattr(
        service_module,
        "evaluate_decision_selection_v1",
        record_decision_selection,
    )
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

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=None,
    )

    results = _result_items(research, run["runCode"])
    assert len(results) == 1
    assert results[0]["disposition"] == "qualified"
    receipts = _search_receipt_items(research, run["runCode"])
    assert len(receipts) == 1
    bundle = receipts[0]["receiptBundle"]
    assert bundle["stopReason"] == "qualified"
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    assert [
        item["attempt"] for item in bundle["evaluationReceipts"]
    ] == ["primary", "qualified_repeat"]
    primary, repeated = bundle["evaluationReceipts"]
    assert primary["candidateIntegrityCore"] == repeated[
        "candidateIntegrityCore"
    ]
    assert primary["evaluation"] == repeated["evaluation"]
    assert bundle["qualifiedCandidateHash"] == results[0]["provenance"][
        "candidateHash"
    ]
    assert len(qualification_calls) == 2
    assert qualification_calls[0] == qualification_calls[1]
    coverage = research.compiled_domain.compiled.coverage
    required_base_failures = {
        "decision_material_counterexample",
        "decision_recommendation_mismatch",
        *(
            ("domain_executable_binding_coverage_incomplete",)
            if not coverage.executable_binding_complete
            else ()
        ),
        *(
            ("domain_behavioral_or_invariant_coverage_unproven",)
            if not coverage.behavioral_or_invariant_complete
            else ()
        ),
    }
    assert required_base_failures <= set(qualification_calls[0]["failures"])
    assert len(decision_selection_calls) == 2
    assert decision_selection_calls[0] == decision_selection_calls[1]
    assert decision_selection_calls[0]["profileHash"] == decision_score_profile.sha256
    assert decision_selection_calls[0]["passed"] is True
    assert decision_selection_calls[0]["evaluatedDecisionCount"] == 25
    assert decision_selection_calls[0]["qualifiedDecisionCount"] > 0
    assert decision_selection_calls[0]["rejectionReasons"] == []
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0
    names = [item[0] for item in call_order]
    assert names.count("receipt") == 1
    assert names.count("result") == 1
    assert names.count("finalize") == 1
    assert names.count("complete") == 1
    assert names.index("receipt") < names.index("result")
    assert names.index("receipt") < names.index("finalize")
    assert names.index("receipt") < names.index("complete")
    receipt_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "receipt"
    )
    result_kwargs = next(
        kwargs for name, _args, kwargs in call_order if name == "result"
    )
    result_args = next(
        args for name, args, _kwargs in call_order if name == "result"
    )
    assert receipt_kwargs == {
        "run_code": run["runCode"],
        "shard_id": claim["shardId"],
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
        "seed": claim["seedStart"],
    }
    assert result_args[0]["provenance"]["researchShardId"] == claim["shardId"]
    assert result_kwargs == {
        "attempt_count": claim["attemptCount"],
        "fencing_token": claim["fencingToken"],
        "worker_id": claim["leaseOwner"],
    }
    shard = research.store.get_shard(claim["shardId"])
    assert shard is not None
    assert shard["result"]["resultCount"] == 1
    assert shard["result"]["qualifiedCount"] == 1
    assert shard["result"]["rejectedCount"] == 0
    assert shard["result"]["quarantineCount"] == 0

    call_order.clear()
    qualification_calls.clear()
    decision_selection_calls.clear()
    secondary = service_harness.simulator.simulate(
        2,
        dealer=Seat.EAST,
        match_score=(0, 0),
    )
    assert secondary.final_trace.trusted_final_state["phase"] in {
        GamePhase.COMPLETE.value,
        GamePhase.SCORED.value,
    }
    run, claim = _prepare_ortools_branch(
        service_harness,
        monkeypatch,
        label="qualified-repeat-source-conflict",
        resources={"analysisMode": "screening"},
    )
    generator = research.deal_generator
    assert isinstance(generator, _SequentialGeneratorStub)
    monkeypatch.setattr(
        research,
        "simulator",
        _SequenceSimulator(
            service_harness.simulator,
            (service_harness.simulation, secondary),
        ),
    )

    research._execute_claimed_shard(
        run["runCode"],
        claim,
        claim["leaseOwner"],
        context=None,
    )

    assert _result_items(research, run["runCode"]) == []
    receipts = _search_receipt_items(research, run["runCode"])
    assert len(receipts) == 1
    bundle = receipts[0]["receiptBundle"]
    assert bundle["stopReason"] == "fatal_quarantine"
    assert [
        item["evaluation"]["outcome"]
        for item in bundle["evaluationReceipts"]
    ] == ["qualified", "fatal_quarantine"]
    assert bundle["rejectionFeedbackRows"] == []
    assert bundle["terminalContinuationReceipt"] is None
    disagreements = research.store.list_disagreements(run_code=run["runCode"])
    assert len(disagreements) == 1
    disagreement = disagreements[0]
    assert disagreement["category"] == "runtime_transition_defect"
    assert disagreement["runtimeEvidence"]["stage"] == (
        "candidate_output_determinism_admission"
    )
    records = _audit_items(research, run["runCode"])
    assert len(records) == 1
    record = records[0]
    assert record["traceHash"] == service_harness.simulation.final_trace.sha256
    assert record["disposition"] == "quarantined"
    assert {
        "stage_qualified_repeat_determinism",
        "runtime_transition_defect",
    }.issubset(record["failureCategories"])
    assert record["disagreementReferences"] == [{
        "disagreementId": disagreement["disagreementId"],
        "sha256": disagreement["sha256"],
        "category": disagreement["category"],
        "status": disagreement["status"],
    }]
    names = [item[0] for item in call_order]
    assert names.count("receipt") == 1
    assert names.count("result") == 0
    assert names.count("disagreement") == 1
    assert names.count("finalize") == 1
    assert names.count("complete") == 1
    assert names.index("receipt") < names.index("disagreement")
    assert names.index("receipt") < names.index("finalize")
    assert names.index("receipt") < names.index("complete")
    assert len(qualification_calls) == 1
    assert len(decision_selection_calls) == 1
    assert decision_selection_calls[0]["profileHash"] == decision_score_profile.sha256
    assert decision_selection_calls[0]["passed"] is True
    assert generator.generate_calls == [()]
    assert generator.probe_calls == []
    assert generator.legacy_generate_calls == 0
