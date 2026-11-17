"""Durable asynchronous control plane for complete natural Sakka generation."""
from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping

from .sakka_generation import NaturalSakkaGeneratorV1


class _JobCancellation:
    def __init__(self, jobs: Any, job_id: str):
        self.jobs = jobs
        self.job_id = job_id

    def __call__(self) -> bool:
        return bool(self.jobs.cancelled(self.job_id))


class NaturalSakkaGenerationJobServiceV1:
    def __init__(
        self,
        *,
        jobs: Any,
        background_executor: Any,
        generator: NaturalSakkaGeneratorV1,
        catalogs: Any,
        maximum_attempts: int,
    ) -> None:
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
        self.jobs = jobs
        self.background_executor = background_executor
        self.generator = generator
        self.catalogs = catalogs
        self.maximum_attempts = maximum_attempts
        self._active_lock = threading.Lock()
        self._active_jobs: set[str] = set()

    def submit(
        self,
        *,
        root_seed: int,
        initial_dealer: str,
        idempotency_key: str,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        seed = int(root_seed)
        if seed < 0 or seed > 2**63 - 1:
            raise ValueError("natural Sakka root seed must be a signed 64-bit integer")
        dealer = str(initial_dealer).strip().lower()
        if dealer not in {"north", "east", "south", "west"}:
            raise ValueError("natural Sakka initial dealer is invalid")
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("natural Sakka generation requires a bounded idempotency key")
        request = {
            "schemaVersion": 1,
            "rootSeed": seed,
            "initialDealer": dealer,
            "profileHash": self.generator.profile.sha256,
        }
        job_id = uuid.uuid4().hex
        actual_id = self.jobs.create(
            job_id,
            request,
            kind="natural_sakka_generation_v1",
            tenant_id=tenant_id,
            max_attempts=self.maximum_attempts,
            idempotency_key=idempotency_key,
        )
        if actual_id == job_id:
            self.background_executor.submit(self._execute, job_id, request)
        result = self.jobs.get(actual_id)
        if result is None:
            raise RuntimeError("natural Sakka generation job was not persisted")
        persisted_request = dict(result.get("request") or {})
        persisted_request.pop("idempotencyKey", None)
        if persisted_request != request:
            raise ValueError("Idempotency-Key was already used for a different generation")
        return result

    def get(self, job_id: str) -> dict[str, Any] | None:
        item = self.jobs.get(job_id)
        if item is None or item.get("kind") != "natural_sakka_generation_v1":
            return None
        return item

    def list(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.jobs.list_jobs(
            kinds=("natural_sakka_generation_v1",), status=status, limit=limit
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
                error="cancelled before natural Sakka generation resumed",
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
                        error="cancelled during natural Sakka generation recovery",
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
                error="cancelled before natural Sakka generation execution",
            )
            return
        self.jobs.update(job_id, status="running", progress=0.03)

        def progress(completed: int, maximum: int) -> None:
            self.jobs.update(
                job_id,
                progress=min(0.82, 0.05 + 0.77 * float(completed) / float(maximum)),
            )

        try:
            result = self.generator.generate(
                int(request["rootSeed"]),
                initial_dealer=str(request["initialDealer"]),
                cancel_check=_JobCancellation(self.jobs, job_id),
                progress_callback=progress,
            )
            self.jobs.update(job_id, progress=0.86)
            persisted = self.catalogs.persist_natural_sakka_generation(result)
            self.jobs.update(
                job_id,
                status="completed",
                progress=1.0,
                result={
                    "schemaVersion": 1,
                    "generationHash": result.sha256,
                    "sakkaHash": result.sakka.sha256,
                    "roundCount": len(result.rounds),
                    "finalScore": list(result.sakka.final_score),
                    "winner": result.sakka.winner,
                    "artifact": persisted["record"]["artifact"],
                },
            )
        except InterruptedError as exc:
            self.jobs.update(job_id, status="cancelled", progress=1.0, error=str(exc))
        except Exception as exc:
            status = "cancelled" if self.jobs.cancelled(job_id) else "failed"
            self.jobs.update(
                job_id,
                status=status,
                progress=1.0,
                error=f"{type(exc).__name__}: {exc}",
            )
