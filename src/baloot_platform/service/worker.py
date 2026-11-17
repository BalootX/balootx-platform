from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from baloot_platform.config import data_root, platform_root

from baloot_platform.storage.jobs import JobStore
from baloot_platform.storage.postgres import PostgresJobStore


def main() -> None:  # pragma: no cover - operational process
    root = platform_root()
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
        if store.cancelled(job["jobId"]):
            store.update(job["jobId"], status="cancelled", worker_id=worker_id)
            continue
        _execute_job(job["jobId"], job["request"], store=store)
