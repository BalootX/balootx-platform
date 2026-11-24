from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from baloot_platform.observability import EVENT_LOGGER, Telemetry
from baloot_platform.service.mcp_server import TOOL_DEFINITIONS, create_http_app


def test_v20_mcp_streamable_get_sse_and_tool_annotations(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BALOOT_ENV", "test"); monkeypatch.setenv("BALOOT_DATA_DIR", str(tmp_path))
    client = TestClient(create_http_app())
    response = client.get("/mcp", headers={"Accept": "text/event-stream"})
    assert response.status_code == 200 and "text/event-stream" in response.headers["content-type"]
    assert "event: endpoint" in response.text and response.headers.get("mcp-session-id")
    assert all(set(item["annotations"]) == {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"} for item in TOOL_DEFINITIONS)


def test_v20_observability_rejects_sensitive_attributes():
    telemetry = Telemetry("v20-test")
    attrs = telemetry.attributes({"game_id": "G", "hidden_cards": "secret", "access_token": "token", "operation_id": "op"})
    assert attrs["baloot.game.id"] == "G" and attrs["baloot.operation.id"] == "op"
    assert not any("hidden" in key or "token" in key for key in attrs)


def _events(caplog):
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == EVENT_LOGGER.name
    ]


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", "/healthz", 200),
        ("GET", "/missing/private-name", 404),
        ("POST", "/api/v1/options", 401),
        ("POST", "/api/v1/options", 413),
    ],
)
def test_http_request_logs_contain_only_server_ids_and_status(
    monkeypatch, caplog, method, path, status
):
    from baloot_platform.service import api

    monkeypatch.setenv("BALOOT_MAX_BODY_BYTES", "1" if status == 413 else "1024")
    monkeypatch.setenv("BALOOT_API_KEY", "synthetic-configured-key")
    monkeypatch.setenv("BALOOT_RATE_LIMIT_PER_MINUTE", "0")
    caplog.set_level(logging.INFO, logger=EVENT_LOGGER.name)
    client = TestClient(api.app)
    response = client.request(
        method,
        f"{path}?ticket=private-query",
        headers={"x-request-id": "private-caller-id", "x-api-key": "private-key"},
        content=b"private-body" if method == "POST" else None,
    )

    assert response.status_code == status
    request_id = response.headers["x-request-id"]
    assert uuid.UUID(request_id).version == 4
    assert _events(caplog) == [{
        "event": "request.response",
        "request_id": request_id,
        "transport": "http",
        "status_code": status,
    }]
    if status in {401, 413}:
        assert response.json()["requestId"] == request_id


def test_http_request_error_is_safe(monkeypatch, caplog):
    from baloot_platform.service import api

    def fail_identity(_token):
        raise RuntimeError("private-exception")

    monkeypatch.setattr(api.IDENTITY, "resolve", fail_identity)
    caplog.set_level(logging.INFO, logger=EVENT_LOGGER.name)
    client = TestClient(api.app, raise_server_exceptions=False)
    response = client.get("/healthz", headers={"x-request-id": "private-caller-id"})
    assert response.status_code == 500
    assert _events(caplog) == [{
        "event": "request.failed",
        "request_id": response.headers["x-request-id"],
        "transport": "http",
        "error_type": "RuntimeError",
    }]


@pytest.mark.parametrize(
    ("query", "status", "close_code"),
    [("access_token=private-token", 403, 4401), ("ticket=private-ticket", 101, 4404)],
)
def test_websocket_handshake_logs_are_safe(monkeypatch, caplog, query, status, close_code):
    from baloot_platform.service import api

    def consume(ticket, *, room_id):
        assert (ticket, room_id) == ("private-ticket", "private-room")
        return SimpleNamespace(subject_id="synthetic-subject")

    monkeypatch.setattr(api.WEBSOCKET_TICKETS, "consume", consume)
    monkeypatch.setattr(api.IDENTITY, "resolve_subject", lambda _: SimpleNamespace(subject_id="synthetic-subject"))
    # An authenticated connection to a missing room closes after the upgrade.
    monkeypatch.delenv("BALOOT_WEBAUTHN_ORIGIN", raising=False)
    caplog.set_level(logging.INFO, logger=EVENT_LOGGER.name)
    client = TestClient(api.app)
    with pytest.raises(WebSocketDisconnect) as disconnected:
        with client.websocket_connect(
            f"/api/v1/live/rooms/private-room/events?{query}",
            headers={"x-request-id": "private-caller-id"},
        ) as socket:
            socket.receive_json()

    assert disconnected.value.code == close_code
    events = _events(caplog)
    assert len(events) == 1
    assert uuid.UUID(events[0].pop("request_id")).version == 4
    assert events[0] == {
        "event": "request.response",
        "transport": "websocket",
        "status_code": status,
    }


def test_api_entrypoint_disables_raw_access_and_redacts_server_errors(monkeypatch):
    from baloot_platform.service import api

    settings = {}
    monkeypatch.setattr(api, "configure_event_logging", lambda: None)
    monkeypatch.setattr(api.uvicorn, "run", lambda _app, **kwargs: settings.update(kwargs))
    api.main()

    assert settings["access_log"] is False
    assert settings["log_level"] == "warning"
    formatter = settings["log_config"]["formatters"]["default"]["()"]()
    error = ValueError("private-body")
    record = logging.LogRecord(
        "uvicorn.error", logging.ERROR, "h11_impl.py", 10, "bad path %s",
        ("private-path",), (type(error), error, None), "run_asgi",
    )
    event = json.loads(formatter.format(record))
    assert uuid.UUID(event.pop("trace_id")).version == 4
    assert event == {
        "event": "server.diagnostic",
        "level": "ERROR",
        "source": "h11_impl.run_asgi:10",
        "error_type": "ValueError",
    }


@pytest.fixture
def worker_logs(monkeypatch, caplog, capsys):
    monkeypatch.setattr(EVENT_LOGGER, "handlers", [])
    monkeypatch.setattr(EVENT_LOGGER, "propagate", True)
    caplog.set_level(logging.INFO, logger=EVENT_LOGGER.name)
    return capsys


def test_video_worker_logs_safe_result_and_exits_on_fatal_error(monkeypatch, worker_logs):
    from baloot_platform.service import worker

    job_id = "0123456789abcdef"
    jobs = iter([{"jobId": job_id, "request": {"samples": "private-job-input"}}])
    updates = []

    def claim(*_args, **_kwargs):
        job = next(jobs, None)
        if job is None:
            raise OSError("private-storage-error")
        return job

    store = SimpleNamespace(
        claim=claim, cancelled=lambda _id: False,
        update=lambda _id, **fields: updates.append(fields),
    )
    monkeypatch.delenv("BALOOT_DATABASE_URL", raising=False)
    monkeypatch.setenv("BALOOT_WORKER_ID", "private-worker-host")
    monkeypatch.setattr(worker, "JobStore", lambda _path: store)
    with pytest.raises(SystemExit) as exit_error:
        worker.main()

    assert exit_error.value.code == 1
    assert updates[0]["status"] == "failed"
    assert "private-job-input" in updates[0]["error"]
    output = worker_logs.readouterr()
    assert output.out == ""
    events = [json.loads(line) for line in output.err.splitlines()]
    assert all(uuid.UUID(event.pop("trace_id")).version == 4 for event in events)
    assert events == [
        {"event": "worker.job_result", "worker": "video", "job_id": job_id, "status": "failed"},
        {"event": "worker.failed", "worker": "video", "error_type": "OSError"},
    ]


def test_foundry_fleet_fatal_log_excludes_worker_identities(monkeypatch, worker_logs):
    from baloot_platform.service import foundry_simulation_fleet as fleet

    monkeypatch.setenv("BALOOT_FOUNDRY_CAMPAIGN_HASH", "a" * 64)
    monkeypatch.setenv("BALOOT_FOUNDRY_WORKER_COUNT", "1")
    monkeypatch.setenv("BALOOT_FOUNDRY_RUN_ID", "synthetic-run")
    monkeypatch.setenv("BALOOT_FOUNDRY_HOST_ID", "synthetic-host")
    monkeypatch.setenv("BALOOT_FOUNDRY_FLEET_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(
        fleet,
        "_campaign_snapshot",
        lambda _: ("running", ("private-worker-host",), ("organic",)),
    )
    with pytest.raises(SystemExit) as exit_error:
        fleet.main()

    assert exit_error.value.code == 1
    output = worker_logs.readouterr()
    assert output.out == ""
    event = json.loads(output.err)
    assert uuid.UUID(event.pop("trace_id")).version == 4
    assert event == {
        "event": "worker.failed",
        "worker": "foundry_simulation_fleet",
        "error_type": "ValueError",
    }


def test_v20_configuration_studio_is_guided_and_scope_aware():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/baloot_platform/service/static/configuration/index.html").read_text(encoding="utf-8")
    for item in ("scopeKey", "environment", "effectiveFrom", "effectiveUntil", "rolloutPercentage", "user_experiment"):
        assert item in text
    assert "JSON يظهر فقط في الوضع المتقدم" in text


def test_v20_semantic_operation_registry_generation(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy(); env.update({"PYTHONPATH": str(root / "src"), "BALOOT_ROOT": str(root), "BALOOT_ENV": "test", "BALOOT_DATA_DIR": str(tmp_path)})
    result = subprocess.run([sys.executable, "scripts/generate_capability_parity_v20.py"], cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads((root / "generated/capability-parity-v20.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed" and payload["semanticParity"] is True and payload["declaredOperations"] >= 14


def test_v20_player_shell_uses_guided_summaries_instead_of_raw_json():
    root = Path(__file__).resolve().parents[1]
    script = (root / "src/baloot_platform/service/static/app/app.js").read_text(encoding="utf-8")
    assert "sharedSummary(data)" in script and "roomSummary(data)" in script
    assert "replaySummary(data)" in script and "caseSummary(data)" in script
    assert "textContent=JSON.stringify(data,null,2)" not in script
