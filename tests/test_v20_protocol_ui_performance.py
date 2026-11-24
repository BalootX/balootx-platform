from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from baloot_platform.observability import Telemetry
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


def test_v20_configuration_studio_is_guided_and_scope_aware():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/baloot_platform/service/static/configuration/index.html").read_text(encoding="utf-8")
    for item in ("scopeKey", "environment", "effectiveFrom", "effectiveUntil", "rolloutPercentage", "user_experiment"):
        assert item in text
    assert 'aria-describedby="timeZoneHelp"' in text
    assert "timeZone: $('timeZone').value.trim()" in text
    assert "localSchedule," in text
    assert ".toISOString()" not in text
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
