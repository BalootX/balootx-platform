"""Durable asynchronous control plane for verified critical-context search."""
from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping, Sequence

from .context_search import CriticalContextSearchV1


class _JobCancellation:
    def __init__(self, jobs: Any, job_id: str):
        self.jobs = jobs
        self.job_id = job_id

    def __call__(self) -> bool:
        return bool(self.jobs.cancelled(self.job_id))


class ContextSearchJobServiceV1:
    """Submit, inspect, and cancel content-addressed context searches."""

    def __init__(
        self,
        *,
        jobs: Any,
        background_executor: Any,
        search_engine: CriticalContextSearchV1,
        catalogs: Any,
        maximum_attempts: int,
    ) -> None:
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
        self.jobs = jobs
        self.background_executor = background_executor
        self.search_engine = search_engine
        self.catalogs = catalogs
        self.maximum_attempts = maximum_attempts
        self._active_lock = threading.Lock()
        self._active_jobs: set[str] = set()

    def submit(
        self,
        *,
        parent_trace_reference: Mapping[str, Any],
        decision_id: str,
        context_proof_hashes: Sequence[str],
        idempotency_key: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if not decision_id or len(decision_id) > 200:
            raise ValueError("context search requires a bounded decision ID")
        if not 1 <= len(context_proof_hashes) <= self.search_engine.profile.maximum_contexts:
            raise ValueError("context search proof count exceeds the governed bound")
        if len(set(context_proof_hashes)) != len(context_proof_hashes):
            raise ValueError("context search proof hashes must be unique")
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("context search requires a bounded idempotency key")
        request = {
            "schemaVersion": 1,
            "parentTraceReference": dict(parent_trace_reference),
            "decisionId": decision_id,
            "contextProofHashes": sorted(str(item) for item in context_proof_hashes),
            "profileHash": self.search_engine.profile.sha256,
        }
        job_id = uuid.uuid4().hex
        actual_id = self.jobs.create(
            job_id,
            request,
            kind="critical_context_search_v1",
            tenant_id=tenant_id,
            max_attempts=self.maximum_attempts,
            idempotency_key=idempotency_key,
        )
        if actual_id == job_id:
            self.background_executor.submit(self._execute, job_id, request)
        result = self.jobs.get(actual_id)
        if result is None:
            raise RuntimeError("context search job was not persisted")
        return result

    def get(self, job_id: str) -> dict[str, Any] | None:
        item = self.jobs.get(job_id)
        if item is None or item.get("kind") != "critical_context_search_v1":
            return None
        return item

    def list(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.jobs.list_jobs(
            kinds=("critical_context_search_v1",), status=status, limit=limit
        )

    def cancel(self, job_id: str) -> bool:
        if self.get(job_id) is None:
            return False
        accepted = bool(self.jobs.request_cancel(job_id))
        if not accepted:
            return False
        with self._active_lock:
            active = job_id in self._active_jobs
        if not active:
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before context search resumed",
            )
        return True

    def events(self, job_id: str) -> list[dict[str, Any]] | None:
        if self.get(job_id) is None:
            return None
        return list(self.jobs.events(job_id))

    def recover_pending(self) -> tuple[str, ...]:
        """Resume local jobs after restart; callers must gate this to single-node mode."""
        recovered: set[str] = set()
        for status in ("queued", "running"):
            for item in self.list(status=status, limit=1_000):
                job_id = str(item["jobId"])
                if item.get("cancelRequested"):
                    self.jobs.update(
                        job_id,
                        status="cancelled",
                        progress=1.0,
                        error="cancelled during context search recovery",
                    )
                else:
                    self.background_executor.submit(
                        self._execute, job_id, dict(item["request"])
                    )
                recovered.add(job_id)
        return tuple(sorted(recovered))

    def _execute(self, job_id: str, request: Mapping[str, Any]) -> None:
        with self._active_lock:
            if job_id in self._active_jobs:
                return
            self._active_jobs.add(job_id)
        try:
            self._execute_once(job_id, request)
        finally:
            with self._active_lock:
                self._active_jobs.discard(job_id)

    def _execute_once(self, job_id: str, request: Mapping[str, Any]) -> None:
        if self.jobs.cancelled(job_id):
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before context search execution",
            )
            return
        self.jobs.update(job_id, status="running", progress=0.05)
        try:
            parent = self.catalogs.load_complete_trace(request["parentTraceReference"])
            contexts = tuple(
                self.catalogs.load_verified_score_context(item)
                for item in request["contextProofHashes"]
            )
            self.jobs.update(job_id, progress=0.15)
            result = self.search_engine.search(
                parent,
                decision_id=str(request["decisionId"]),
                contexts=contexts,
                cancel_check=_JobCancellation(self.jobs, job_id),
            )
            persisted = self.catalogs.persist_context_search_result(result)
            self.jobs.update(
                job_id,
                status="completed",
                progress=1.0,
                result={
                    "schemaVersion": 1,
                    "searchResultHash": result.sha256,
                    "selectedHash": result.selected_hash,
                    "evaluatedCount": len(result.evaluated),
                    "acceptedCount": len(result.accepted),
                    "artifact": persisted["record"]["artifact"],
                },
            )
        except InterruptedError as exc:
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error=str(exc),
            )
        except Exception as exc:
            status = "cancelled" if self.jobs.cancelled(job_id) else "failed"
            self.jobs.update(
                job_id,
                status=status,
                progress=1.0,
                error=f"{type(exc).__name__}: {exc}",
            )
