from __future__ import annotations

import copy
from typing import Any

import pytest


from research_audit_trace_helpers import (
    _Harness,
    _audit_items,
    _claim_case,
    _payload,
    _result_items,
    _search_receipt_items,
)
from research_audit_trace_helpers import service_harness as service_harness


def test_retry_rejects_tampered_run_generator_identity_before_heartbeat(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, first_claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="tampered-generator-retry",
        payload=_payload(
            research,
            label="tampered-generator-retry",
            lane="ortools",
        ),
    )
    research.store.fail_shard(
        first_claim["shardId"],
        worker_id=first_claim["leaseOwner"],
        fencing_token=first_claim["fencingToken"],
        error={"type": "fixture_retry", "message": "retry identity check"},
        retryable=True,
    )
    retry_claim = research.store.claim_shard(
        run["runCode"],
        worker_id="audit-service-worker:tampered-generator-retry-2",
        lease_seconds=30,
    )
    assert retry_claim is not None
    assert retry_claim["attemptCount"] == 2
    tampered_identity = copy.deepcopy(run["dealGeneratorIdentity"])
    tampered_identity["sha256"] = "0" * 64
    research.store.update_run(
        run["runCode"],
        dealGeneratorIdentity=tampered_identity,
    )
    heartbeat_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_heartbeat = research.store.heartbeat_shard

    def heartbeat_spy(*args: Any, **kwargs: Any) -> Any:
        heartbeat_calls.append((args, copy.deepcopy(kwargs)))
        return original_heartbeat(*args, **kwargs)

    monkeypatch.setattr(research.store, "heartbeat_shard", heartbeat_spy)

    with pytest.raises(ValueError, match="deal-generator identity mismatch"):
        research._execute_claimed_shard(
            run["runCode"],
            retry_claim,
            retry_claim["leaseOwner"],
            context=None,
        )

    assert heartbeat_calls == []
    assert _search_receipt_items(
        research,
        run["runCode"],
        admitted_only=False,
    ) == []


def test_retry_after_result_put_reuses_identical_result_payload(
    service_harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = service_harness.research
    run, first_claim, _ = _claim_case(
        service_harness,
        monkeypatch,
        label="retry-after-result-put",
        payload=_payload(
            research,
            label="retry-after-result-put",
            lane="organic_runtime",
        ),
    )
    monkeypatch.setattr(research, "_semantic_mismatches", lambda *_a, **_k: [])
    result_payloads: list[dict[str, Any]] = []
    completion_calls = 0
    original_put_result = research.store.put_result
    original_complete_shard = research.store.complete_shard

    def capture_result(payload: dict[str, Any], **kwargs: Any) -> Any:
        result_payloads.append(copy.deepcopy(payload))
        return original_put_result(payload, **kwargs)

    def interrupt_first_completion(*args: Any, **kwargs: Any) -> Any:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            raise InterruptedError("fixture interruption after result put")
        return original_complete_shard(*args, **kwargs)

    monkeypatch.setattr(research.store, "put_result", capture_result)
    monkeypatch.setattr(
        research.store,
        "complete_shard",
        interrupt_first_completion,
    )

    research._execute_claimed_shard(
        run["runCode"],
        first_claim,
        first_claim["leaseOwner"],
        context=None,
    )

    interrupted = research.store.get_shard(first_claim["shardId"])
    assert interrupted is not None
    assert interrupted["status"] == "queued"
    assert interrupted["retryCount"] == 1
    assert len(result_payloads) == 1
    assert result_payloads[0]["createdAt"] == run["createdAt"]
    assert research.get_result(result_payloads[0]["resultCode"]) is None

    retry_claim = research.store.claim_shard(
        run["runCode"],
        worker_id="audit-service-worker:retry-after-result-put-2",
        lease_seconds=30,
    )
    assert retry_claim is not None
    assert retry_claim["attemptCount"] == first_claim["attemptCount"] + 1
    assert retry_claim["fencingToken"] == first_claim["fencingToken"] + 1

    research._execute_claimed_shard(
        run["runCode"],
        retry_claim,
        retry_claim["leaseOwner"],
        context=None,
    )

    completed = research.store.get_shard(retry_claim["shardId"])
    assert completed is not None and completed["status"] == "completed"
    assert completion_calls == 2
    assert len(result_payloads) == 2
    assert result_payloads[0] == result_payloads[1]
    admitted = _result_items(research, run["runCode"])
    assert admitted == [result_payloads[0]]
    records = _audit_items(research, run["runCode"])
    assert len(records) == 2
    assert {
        int(record["source"]["attemptCount"]): record["attemptStatus"]
        for record in records
    } == {
        first_claim["attemptCount"]: "retry_scheduled",
        retry_claim["attemptCount"]: "completed",
    }
