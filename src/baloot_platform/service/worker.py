from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from baloot_platform.config import data_root
from baloot_platform.observability import log_worker_result, run_worker

from baloot_platform.storage.jobs import JobStore
from baloot_platform.storage.postgres import PostgresJobStore


def _run() -> None:
    runtime = Path(os.environ.get("BALOOT_RUNTIME", data_root() / "runtime")).expanduser().resolve()
    database_url = os.environ.get("BALOOT_DATABASE_URL", "").strip()
    store = PostgresJobStore(database_url) if database_url else JobStore(runtime / "jobs.sqlite3")
    worker_id = os.environ.get("BALOOT_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")
    from baloot_platform.service.api import _execute_job
    while True:
        job = store.claim(worker_id, kinds=("video",), lease_seconds=120)
        if job is None:
            time.sleep(float(os.environ.get("BALOOT_WORKER_POLL_SECONDS", "0.5")))
            continue
        job_id = job["jobId"]
        if store.cancelled(job_id):
            cancelled = store.update(job_id, status="cancelled", worker_id=worker_id)
            status = "cancelled" if cancelled else "lease_lost"
        else:
            status = _execute_job(job_id, job["request"], store=store)
        log_worker_result("video", job_id, status)


def main() -> None:
    run_worker("video", _run)
