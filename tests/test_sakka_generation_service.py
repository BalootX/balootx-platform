from __future__ import annotations

from types import SimpleNamespace

from baloot_platform.challenge.sakka_generation_service import (
    NaturalSakkaGenerationJobServiceV1,
)
from baloot_platform.storage.jobs import JobStore


class InlineExecutor:
    def submit(self, operation, *args):
        operation(*args)


class DeferredExecutor:
    def __init__(self):
        self.tasks = []

    def submit(self, operation, *args):
        self.tasks.append((operation, args))


class Generator:
    profile = SimpleNamespace(sha256="a" * 64)

    def generate(self, root_seed, *, initial_dealer, cancel_check, progress_callback):
        assert root_seed == 20260803
        assert initial_dealer == "east"
        assert not cancel_check()
        progress_callback(2, 10)
        return SimpleNamespace(
            sha256="b" * 64,
            sakka=SimpleNamespace(
                sha256="c" * 64,
                final_score=(152, 140),
                winner="north_south",
            ),
            rounds=(object(), object()),
        )


class Catalogs:
    def __init__(self):
        self.persisted = []

    def persist_natural_sakka_generation(self, result):
        self.persisted.append(result.sha256)
        return {"record": {"artifact": {"sha256": "d" * 64}}}


def test_natural_sakka_generation_job_is_idempotent_and_persisted(tmp_path):
    catalogs = Catalogs()
    service = NaturalSakkaGenerationJobServiceV1(
        jobs=JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=InlineExecutor(),
        generator=Generator(),
        catalogs=catalogs,
        maximum_attempts=3,
    )
    first = service.submit(
        root_seed=20260803,
        initial_dealer="east",
        idempotency_key="same-natural-sakka",
    )
    second = service.submit(
        root_seed=20260803,
        initial_dealer="east",
        idempotency_key="same-natural-sakka",
    )

    assert first["jobId"] == second["jobId"]
    assert second["status"] == "completed"
    assert second["result"]["generationHash"] == "b" * 64
    assert second["result"]["sakkaHash"] == "c" * 64
    assert catalogs.persisted == ["b" * 64]
    assert service.events(first["jobId"])[-1]["type"] == "completed"
    try:
        service.submit(
            root_seed=20260804,
            initial_dealer="east",
            idempotency_key="same-natural-sakka",
        )
    except ValueError as exc:
        assert "different generation" in str(exc)
    else:
        raise AssertionError("idempotency key reuse with a different seed was accepted")


def test_natural_sakka_generation_rejects_invalid_inputs(tmp_path):
    service = NaturalSakkaGenerationJobServiceV1(
        jobs=JobStore(tmp_path / "jobs.sqlite3"),
        background_executor=InlineExecutor(),
        generator=Generator(),
        catalogs=Catalogs(),
        maximum_attempts=3,
    )
    for values in (
        {"root_seed": -1, "initial_dealer": "east", "idempotency_key": "x"},
        {"root_seed": 1, "initial_dealer": "invalid", "idempotency_key": "x"},
        {"root_seed": 1, "initial_dealer": "east", "idempotency_key": ""},
    ):
        try:
            service.submit(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid generation input was accepted: {values}")


def test_natural_sakka_recovers_after_restart_and_terminalizes_inactive_cancel(tmp_path):
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    original = NaturalSakkaGenerationJobServiceV1(
        jobs=jobs,
        background_executor=DeferredExecutor(),
        generator=Generator(),
        catalogs=Catalogs(),
        maximum_attempts=3,
    )
    interrupted = original.submit(
        root_seed=20260803,
        initial_dealer="east",
        idempotency_key="restart-natural-sakka",
    )
    jobs.update(interrupted["jobId"], status="running", progress=0.2)

    restarted = NaturalSakkaGenerationJobServiceV1(
        jobs=jobs,
        background_executor=InlineExecutor(),
        generator=Generator(),
        catalogs=Catalogs(),
        maximum_attempts=3,
    )
    assert restarted.recover_pending() == (interrupted["jobId"],)
    assert restarted.get(interrupted["jobId"])["status"] == "completed"

    stale = original.submit(
        root_seed=20260803,
        initial_dealer="east",
        idempotency_key="cancel-natural-sakka",
    )
    jobs.update(stale["jobId"], status="running", progress=0.2)
    assert restarted.cancel(stale["jobId"])
    terminal = restarted.get(stale["jobId"])
    assert terminal["status"] == "cancelled"
    assert terminal["progress"] == 1.0
