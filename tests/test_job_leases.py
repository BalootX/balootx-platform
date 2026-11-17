from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from baloot_platform.storage.jobs import JobStore


@pytest.mark.parametrize(
    ("stale_owner", "stale_token"),
    [("expired-worker", None), ("current-worker", 1)],
)
def test_reclaimed_job_rejects_stale_worker_events(tmp_path, stale_owner, stale_token):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("video", {"seed": 7})
    with patch("baloot_platform.storage.jobs.time.time", return_value=100):
        store.claim("expired-worker", lease_seconds=1)
    with patch("baloot_platform.storage.jobs.time.time", return_value=102):
        reclaimed = store.claim("current-worker", lease_seconds=60)
    assert reclaimed["leaseOwner"] == "current-worker"
    assert reclaimed["attempts"] == 2
    events = store.events("video")

    store.update(
        "video", status="completed", progress=1.0,
        result={"video": "stale.mp4"}, worker_id=stale_owner,
        fencing_token=stale_token,
    )

    assert store.get("video") == reclaimed
    assert store.events("video") == events

    store.update(
        "video", status="completed", progress=1.0,
        result={"video": "current.mp4"}, worker_id="current-worker",
    )
    completed = store.get("video")
    assert completed["status"] == "completed"
    assert completed["result"] == {"video": "current.mp4"}
    assert completed["leaseOwner"] is None
    completed_events = store.events("video")
    assert completed_events[:-1] == events
    assert completed_events[-1]["type"] == "completed"
    assert completed_events[-1]["payload"] == {"progress": 1.0}


def test_job_update_and_event_roll_back_together(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    store = JobStore(path)
    store.create("video", {})
    store.claim("worker")
    before = store.get("video")
    events = store.events("video")
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TRIGGER reject_completion_event
            BEFORE INSERT ON job_events WHEN NEW.type='completed'
            BEGIN SELECT RAISE(ABORT, 'event write rejected'); END
        """)

    with pytest.raises(sqlite3.IntegrityError, match="event write rejected"):
        store.update("video", status="completed", worker_id="worker")

    assert store.get("video") == before
    assert store.events("video") == events
