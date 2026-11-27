from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from baloot_platform.service.mcp_server import MCPProtocol, create_http_app


def test_mcp_protocol_lists_and_calls_local_tools():
    protocol = MCPProtocol()
    initialized = protocol.process({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    })
    assert initialized["result"]["protocolVersion"] == "2025-11-25"
    listed = protocol.process({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [item["name"] for item in listed["result"]["tools"]]
    again = protocol.process({"jsonrpc": "2.0", "id": 22, "method": "tools/list"})
    assert names == [item["name"] for item in again["result"]["tools"]]
    assert "get_current_bundle" in names and "create_video" in names
    assert {"audit_game", "explain_decision", "record_decision_attempt", "get_video_timeline", "get_video_audit_map", "render_game_replay", "fork_decision", "adaptive_next_challenge"} <= set(names)
    assert {
        "start_challenge_publication_review",
        "get_challenge_publication_reviews",
        "submit_challenge_publication_review",
        "approve_challenge_publication",
        "publish_challenge",
        "retire_challenge",
    } <= set(names)
    assert {"list_sakka_boards", "create_sakka_board", "create_sakka_room", "start_sakka_run", "play_sakka_action", "audit_sakka_run", "share_sakka_decision"} <= set(names)
    assert {"create_render_batch", "get_render_batch", "cancel_render_batch"} <= set(names)
    assert {"create_context_search", "get_context_search", "cancel_context_search"} <= set(names)
    assert {
        "create_natural_sakka_generation",
        "get_natural_sakka_generation",
        "cancel_natural_sakka_generation",
    } <= set(names)
    assert {
        "list_foundry_sakkas",
        "create_natural_sakka",
        "create_curated_sakka",
        "get_foundry_sakka",
    } <= set(names)
    called = protocol.process({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "get_current_bundle", "arguments": {}},
    })
    assert called["result"]["isError"] is False
    assert len(called["result"]["structuredContent"]["sha256"]) == 64
    resources = protocol.process({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
    assert {item["uri"] for item in resources["result"]["resources"]} == {
        "baloot://bundle/current",
        "baloot://rules/current",
        "baloot://agents/profiles",
        "baloot://learning/status",
        "baloot://audit/profile",
        "baloot://difficulty/profile",
        "baloot://sakka/profile",
        "baloot://shared/profile",
        "baloot://live/profile",
        "baloot://adjudication/profile",
        "baloot://security/capabilities",
        "baloot://history/canonical",
        "baloot://hitl/profile",
        "baloot://configuration/registry",
        "baloot://plugins/policy",
        "baloot://performance/profile",
    }


def test_mcp_render_batch_uses_governed_api_and_idempotency_header(monkeypatch):
    from baloot_platform.security import Scope
    from baloot_platform.service import mcp_server

    calls = []

    def fake_api(method, path, payload=None, *, extra_headers=None):
        calls.append((method, path, payload, extra_headers))
        return {"jobId": "RB-1", "status": "queued"}

    monkeypatch.setenv("BALOOT_MCP_ALLOW_MUTATIONS", "true")
    monkeypatch.setattr(mcp_server, "_api", fake_api)
    scope_token = mcp_server._REQUEST_SCOPES.set(
        frozenset({Scope.PLAY.value, Scope.HIDDEN_READ.value})
    )
    try:
        result = mcp_server._call_tool(
            "create_render_batch",
            {
                "idempotency_key": "same-request",
                "items": [{"basename": "decision", "bundleArtifact": {}}],
            },
        )
    finally:
        mcp_server._REQUEST_SCOPES.reset(scope_token)

    assert result["jobId"] == "RB-1"
    assert calls == [(
        "POST",
        "/api/v2/render-batches",
        {"items": [{"basename": "decision", "bundleArtifact": {}}]},
        {"idempotency-key": "same-request"},
    )]


def test_mcp_legacy_video_uses_v4_adapter_and_idempotency_header(monkeypatch):
    from baloot_platform.security import Scope
    from baloot_platform.service import mcp_server

    calls = []

    def fake_api(method, path, payload=None, *, extra_headers=None):
        calls.append((method, path, payload, extra_headers))
        return {
            "jobId": "RUN-ABCDEF123456",
            "status": "queued",
            "authority": "research_v4",
            "rendersAutomatically": False,
        }

    monkeypatch.setenv("BALOOT_MCP_ALLOW_MUTATIONS", "true")
    monkeypatch.setenv("BALOOT_MCP_ALLOW_TRUSTED_CONTEXT", "true")
    monkeypatch.setattr(mcp_server, "_api", fake_api)
    scope_token = mcp_server._REQUEST_SCOPES.set(
        frozenset({Scope.PLAY.value, Scope.HIDDEN_READ.value})
    )
    request = {
        "bundleRevisionId": "CBR-FROZEN-TEST",
        "contract": "sun",
    }
    try:
        result = mcp_server._call_tool(
            "create_video",
            {
                "idempotency_key": "same-video-research",
                "request": request,
            },
        )
    finally:
        mcp_server._REQUEST_SCOPES.reset(scope_token)

    assert result["authority"] == "research_v4"
    assert result["rendersAutomatically"] is False
    assert calls == [
        (
            "POST",
            "/api/v1/video-jobs",
            request,
            {"idempotency-key": "same-video-research"},
        )
    ]


def test_mcp_context_search_uses_governed_api_and_idempotency_header(monkeypatch):
    from baloot_platform.security import Scope
    from baloot_platform.service import mcp_server

    calls = []

    def fake_api(method, path, payload=None, *, extra_headers=None):
        calls.append((method, path, payload, extra_headers))
        return {"jobId": "CTX-1", "status": "queued"}

    monkeypatch.setenv("BALOOT_MCP_ALLOW_MUTATIONS", "true")
    monkeypatch.setattr(mcp_server, "_api", fake_api)
    scope_token = mcp_server._REQUEST_SCOPES.set(
        frozenset({Scope.PLAY.value, Scope.HIDDEN_READ.value})
    )
    try:
        result = mcp_server._call_tool(
            "create_context_search",
            {
                "idempotency_key": "same-context",
                "parent_trace_reference": {"ownerHash": "a" * 64},
                "decision_id": "DEC-1",
                "context_proof_hashes": ["b" * 64],
            },
        )
    finally:
        mcp_server._REQUEST_SCOPES.reset(scope_token)

    assert result["jobId"] == "CTX-1"
    assert calls == [(
        "POST",
        "/api/v2/context-searches",
        {
            "parentTraceReference": {"ownerHash": "a" * 64},
            "decisionId": "DEC-1",
            "contextProofHashes": ["b" * 64],
        },
        {"idempotency-key": "same-context"},
    )]


def test_mcp_natural_sakka_generation_uses_governed_api(monkeypatch):
    from baloot_platform.security import Scope
    from baloot_platform.service import mcp_server

    calls = []

    def fake_api(method, path, payload=None, *, extra_headers=None):
        calls.append((method, path, payload, extra_headers))
        return {"jobId": "SAKKA-GEN-1", "status": "queued"}

    monkeypatch.setenv("BALOOT_MCP_ALLOW_MUTATIONS", "true")
    monkeypatch.setattr(mcp_server, "_api", fake_api)
    scope_token = mcp_server._REQUEST_SCOPES.set(
        frozenset({Scope.PLAY.value, Scope.HIDDEN_READ.value})
    )
    try:
        result = mcp_server._call_tool(
            "create_natural_sakka_generation",
            {
                "root_seed": 20260803,
                "initial_dealer": "east",
                "idempotency_key": "same-sakka",
            },
        )
    finally:
        mcp_server._REQUEST_SCOPES.reset(scope_token)

    assert result["jobId"] == "SAKKA-GEN-1"
    assert calls == [(
        "POST",
        "/api/v2/sakka-generations",
        {"rootSeed": 20260803, "initialDealer": "east"},
        {"idempotency-key": "same-sakka"},
    )]


def test_mcp_stdio_process(root: Path):
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_current_rules", "arguments": {}}},
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["BALOOT_ROOT"] = str(root)
    completed = subprocess.run(
        [sys.executable, "-m", "baloot_platform.service.mcp_server"],
        input="".join(json.dumps(item) + "\n" for item in messages),
        text=True,
        capture_output=True,
        cwd=root,
        env=env,
        timeout=30,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert [item["id"] for item in responses] == [1, 2]
    assert responses[1]["result"]["structuredContent"]["profile"]["status"] == "researched_default_unratified"


def test_mcp_streamable_http_json_mode(monkeypatch):
    monkeypatch.setenv("BALOOT_MCP_AUTH_TOKEN", "secret")
    client = TestClient(create_http_app())
    request = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
    }
    assert client.post("/mcp", json=request).status_code == 401
    response = client.post(
        "/mcp", json=request,
        headers={"Authorization": "Bearer secret", "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-11-25"},
    )
    assert response.status_code == 200
    from baloot_platform.version import __version__
    assert response.json()["result"]["serverInfo"]["version"] == __version__
    stream = client.get("/mcp", headers={"Authorization": "Bearer secret", "Accept": "text/event-stream"})
    assert stream.status_code == 200
    assert "text/event-stream" in stream.headers.get("content-type", "")


def test_mcp_starts_without_oidc_provider_secrets(monkeypatch):
    monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
    monkeypatch.setenv("BALOOT_OIDC_ISSUER", "https://signin.example")
    monkeypatch.delenv("BALOOT_OIDC_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("BALOOT_OIDC_CLIENT_SECRET_EXAMPLE", raising=False)
    monkeypatch.delenv("BALOOT_MCP_AUTHORIZATION_SERVERS", raising=False)
    metadata_path = "/.well-known/oauth-protected-resource"

    with TestClient(create_http_app()) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get(metadata_path).json()["authorization_servers"] == []

    monkeypatch.setenv("BALOOT_MCP_AUTHORIZATION_SERVERS", "https://mcp.example")
    with TestClient(create_http_app()) as client:
        assert client.get(metadata_path).json()["authorization_servers"] == ["https://mcp.example"]
        assert client.get(metadata_path + "/mcp").json()["authorization_servers"] == ["https://mcp.example"]
