from __future__ import annotations

from types import SimpleNamespace

from baloot_platform.challenge.context_search_service import ContextSearchJobServiceV1
from baloot_platform.storage.jobs import JobStore


class InlineExecutor:
    def submit(self, operation, *args):
        operation(*args)


class DeferredExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, operation, *args):
        self.tasks.append((operation, args))


class Catalogs:
    def __init__(self):
        self.persisted = []

    def load_complete_trace(self, reference):
        return {"trace": reference["ownerHash"]}

    def load_verified_score_context(self, proof_hash):
        return {"proof": proof_hash}

    def persist_context_search_result(self, result):
        self.persisted.append(result.sha256)
        return {"record": {"artifact": {"sha256": "b" * 64}}}


class SearchEngine:
    profile = SimpleNamespace(maximum_contexts=10, sha256="p" * 64)

    def search(self, parent, *, decision_id, contexts, cancel_check):
        assert parent["trace"] == "a" * 64
        assert decision_id == "DEC-1"
        assert contexts == ({"proof": "c" * 64},)
        assert not cancel_check()
        return SimpleNamespace(
            sha256="d" * 64,
            selected_hash="e" * 64,
            evaluated=({"accepted": True},),
            accepted=(object(),),
            descendant_traces=(),
        )


def reference():
    return {
        "ownerKind": "complete_trace_v1",
        "ownerHash": "a" * 64,
        "uri": "cas://sha256/" + "b" * 64,
        "sha256": "b" * 64,
        "bytes": 100,
        "mediaType": "application/vnd.balootx.complete-trace-v1+json",
    }


def test_context_search_job_is_idempotent_and_persists_result(tmp_path):
    catalogs = Catalogs()
    service = ContextSearchJobServiceV1(
        jobs=JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=InlineExecutor(),
        search_engine=SearchEngine(),
        catalogs=catalogs,
        maximum_attempts=3,
    )

    first = service.submit(
        parent_trace_reference=reference(),
        decision_id="DEC-1",
        context_proof_hashes=["c" * 64],
        idempotency_key="same",
    )
    second = service.submit(
        parent_trace_reference=reference(),
        decision_id="DEC-1",
        context_proof_hashes=["c" * 64],
        idempotency_key="same",
    )

    assert first["jobId"] == second["jobId"]
    assert second["status"] == "completed"
    assert second["result"]["searchResultHash"] == "d" * 64
    assert catalogs.persisted == ["d" * 64]
    assert service.events(first["jobId"])[-1]["type"] == "completed"


def test_context_search_rejects_duplicate_proofs(tmp_path):
    service = ContextSearchJobServiceV1(
        jobs=JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=InlineExecutor(),
        search_engine=SearchEngine(),
        catalogs=Catalogs(),
        maximum_attempts=3,
    )
    try:
        service.submit(
            parent_trace_reference=reference(),
            decision_id="DEC-1",
            context_proof_hashes=["c" * 64, "c" * 64],
            idempotency_key="duplicate",
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate proof hashes were accepted")


def test_context_search_recovers_after_restart_and_terminalizes_inactive_cancel(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    original = ContextSearchJobServiceV1(
        jobs=jobs,
        background_executor=DeferredExecutor(),
        search_engine=SearchEngine(),
        catalogs=Catalogs(),
        maximum_attempts=3,
    )
    interrupted = original.submit(
        parent_trace_reference=reference(),
        decision_id="DEC-1",
        context_proof_hashes=["c" * 64],
        idempotency_key="restart-context",
    )
    jobs.update(interrupted["jobId"], status="running", progress=0.2)

    restarted = ContextSearchJobServiceV1(
        jobs=jobs,
        background_executor=InlineExecutor(),
        search_engine=SearchEngine(),
        catalogs=Catalogs(),
        maximum_attempts=3,
    )
    assert restarted.recover_pending() == (interrupted["jobId"],)
    assert restarted.get(interrupted["jobId"])["status"] == "completed"

    stale = original.submit(
        parent_trace_reference=reference(),
        decision_id="DEC-1",
        context_proof_hashes=["c" * 64],
        idempotency_key="cancel-context",
    )
    jobs.update(stale["jobId"], status="running", progress=0.2)
    assert restarted.cancel(stale["jobId"])
    terminal = restarted.get(stale["jobId"])
    assert terminal["status"] == "cancelled"
    assert terminal["progress"] == 1.0
