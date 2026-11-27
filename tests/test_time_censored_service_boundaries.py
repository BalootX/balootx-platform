from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from service_test_client import OperatorClient as TestClient
from pydantic import BaseModel

from baloot_platform.analysis.search import SearchTimeBudgetExceeded
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore


def test_search_error_compatibility_imports_share_one_type_and_guard():
    from baloot_platform.analysis import search, search_errors

    assert search.SearchTimeBudgetExceeded is search_errors.SearchTimeBudgetExceeded
    assert search.require_uncensored_play_trace is search_errors.require_uncensored_play_trace


@pytest.fixture(scope="module")
def narrow_api(root):
    # Execute the real handlers without booting unrelated application services.
    source = root / "src/baloot_platform/service/api.py"
    names = {
        "search_time_budget_exceeded", "get_laffa_decision_ledger", "get_laffa_complete_trace",
        "ResearchRenderPrepareRequest", "prepare_any_laffa_render", "_foundry_laffa_evidence",
        "get_laffa_evidence", "get_laffa_scenario_graph",
    }
    nodes = [copy.deepcopy(n) for n in ast.parse(source.read_text(encoding="utf-8")).body
             if getattr(n, "name", None) in names]
    for node in nodes:
        if node.name == "prepare_any_laffa_render":
            node.decorator_list = []
    namespace = {
        "app": FastAPI(), "Request": Request, "HTTPException": HTTPException,
        "JSONResponse": JSONResponse, "SearchTimeBudgetExceeded": SearchTimeBudgetExceeded,
        "BaseModel": BaseModel, "Literal": Literal, "Any": Any,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec", dont_inherit=True), namespace)
    namespace["app"].post("/api/v2/laffat/{identifier}:prepare-render", response_model=None)(
        namespace["prepare_any_laffa_render"]
    )
    return namespace


@pytest.mark.parametrize("path", ["decisions", "complete-trace", "evidence", "scenario-graph"])
def test_trusted_laffa_returns_structured_time_censored_refusal(narrow_api, monkeypatch, path):
    fail = Mock(side_effect=SearchTimeBudgetExceeded("censored live policy"))
    monkeypatch.setitem(narrow_api, "_require_trusted", lambda _request: None)
    monkeypatch.setitem(narrow_api, "LEARNING_STORE", SimpleNamespace(
        decision_ledger=fail, complete_game_trace=fail, list_game_audits=lambda *_a, **_k: [{}],
    ))
    monkeypatch.setitem(narrow_api, "REGISTRY", SimpleNamespace(
        get=lambda *_args: SimpleNamespace(analyze_laffa=fail),
    ))
    with TestClient(narrow_api["app"], raise_server_exceptions=False) as client:
        response = client.get(f"/api/v2/laffat/test/{path}")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"reason": "time_censored", "message": "censored live policy"}
    fail.assert_called_once()


@pytest.mark.parametrize("source", ["laffa", "sakka"])
def test_render_prepare_refuses_censored_laffa_and_sakka(narrow_api, monkeypatch, source):
    fail = Mock(side_effect=SearchTimeBudgetExceeded("censored render source"))
    monkeypatch.setitem(narrow_api, "_require_trusted", lambda _request: None)
    monkeypatch.setitem(narrow_api, "RESEARCH", SimpleNamespace(get_result=lambda _id: None))
    monkeypatch.setitem(narrow_api, "LAFFA_LIBRARY", SimpleNamespace(
        resolve_audit_trace=lambda _id: None,
        resolve_sakka_round=lambda _id: {"traceReference": {"uri": "test"}, "code": "sakka"}
        if source == "sakka" else None,
    ))
    monkeypatch.setitem(narrow_api, "RESEARCH_RENDERS", SimpleNamespace(
        prepare_trace_reference=fail, prepare_laffa=fail,
    ))
    with TestClient(narrow_api["app"]) as client:
        response = client.post("/api/v2/laffat/test:prepare-render", json={})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["reason"] == "time_censored"
    fail.assert_called_once()


def test_censor_handler_preserves_trusted_access_guard(narrow_api, monkeypatch):
    def deny(_request):
        raise HTTPException(403, "trusted access required")

    fail = Mock(side_effect=SearchTimeBudgetExceeded("must not be reached"))
    monkeypatch.setitem(narrow_api, "_require_trusted", deny)
    monkeypatch.setitem(narrow_api, "LEARNING_STORE", SimpleNamespace(decision_ledger=fail))
    with TestClient(narrow_api["app"]) as client:
        assert client.get("/api/v2/laffat/test/decisions").status_code == 403
    fail.assert_not_called()


@pytest.mark.parametrize("path", ["decisions", "complete-trace"])
def test_uncensored_laffa_routes_still_succeed(narrow_api, monkeypatch, path):
    monkeypatch.setitem(narrow_api, "_require_trusted", lambda _request: None)
    value = SimpleNamespace(records=(), to_dict=lambda **_kwargs: {"gameId": "test"})
    monkeypatch.setitem(narrow_api, "LEARNING_STORE", SimpleNamespace(
        decision_ledger=lambda _id: value, complete_game_trace=lambda _id: value,
    ))
    with TestClient(narrow_api["app"]) as client:
        assert client.get(f"/api/v2/laffat/test/{path}").status_code == 200


@pytest.mark.parametrize("transport", ["local", "http"])
def test_mcp_censored_evidence_is_a_structured_tool_error(monkeypatch, transport):
    from baloot_platform.service import mcp_server as mcp

    detail = {"reason": "time_censored", "message": "censored live policy"}
    with httpx.Client(base_url="http://test-api", transport=httpx.MockTransport(
        lambda _request: httpx.Response(409, json={"detail": detail}),
    )) as client:
        monkeypatch.setenv("BALOOT_API_URL", "http://test-api")
        monkeypatch.setattr(mcp, "_API_CLIENTS", {"http://test-api": client})
        call = Mock(side_effect=SearchTimeBudgetExceeded(detail["message"]))
        if transport == "http":
            call = lambda *_args: mcp._api("GET", "/api/v2/laffat/test/complete-trace")
        monkeypatch.setattr(mcp, "_call_tool", call)
        protocol = mcp.MCPProtocol()
        protocol.process({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": mcp.LATEST_PROTOCOL_VERSION, "capabilities": {},
                       "clientInfo": {"name": "test", "version": "1"}},
        })
        response = protocol.process({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_current_bundle", "arguments": {}},
        })
    assert response["result"]["isError"] is True
    assert response["result"]["structuredContent"] == detail
    assert json.loads(response["result"]["content"][0]["text"]) == detail


@pytest.mark.parametrize("body", [b"not json", b'[{"reason":"other"}]', b'{"detail":"other conflict"}'])
def test_mcp_other_http_errors_keep_the_existing_error_behavior(monkeypatch, body):
    from baloot_platform.service import mcp_server as mcp

    with httpx.Client(base_url="http://test-api", transport=httpx.MockTransport(
        lambda _request: httpx.Response(409, content=body),
    )) as client:
        monkeypatch.setenv("BALOOT_API_URL", "http://test-api")
        monkeypatch.setattr(mcp, "_API_CLIENTS", {"http://test-api": client})
        with pytest.raises(httpx.HTTPStatusError):
            mcp._api("GET", "/test")


def test_mcp_stdio_protocol_boundary_keeps_structured_censor_reason(monkeypatch):
    from baloot_platform.service import mcp_server as mcp

    monkeypatch.setattr(mcp.MCPProtocol, "process", Mock(side_effect=SearchTimeBudgetExceeded("censored evidence")))
    output = io.StringIO()
    with monkeypatch.context() as streams:
        streams.setattr(mcp.sys, "stdin", io.StringIO('{"jsonrpc":"2.0","id":1,"method":"resources/read"}\n'))
        streams.setattr(mcp.sys, "stdout", output)
        mcp.run_stdio()
    response = json.loads(output.getvalue())
    assert response["error"]["code"] == -32000
    assert response["error"]["data"]["reason"] == "time_censored"


@pytest.mark.real_clock
def test_grpc_censor_interceptor_returns_failed_precondition(root):
    import grpc

    source = root / "src/baloot_platform/service/grpc_service.py"
    serve = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.AsyncFunctionDef) and n.name == "serve")
    interceptor = next(n for n in serve.body if isinstance(n, ast.ClassDef) and n.name == "TimeCensoredInterceptor")
    server_assignment = next(n for n in serve.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == "server" for t in n.targets))
    assert any(isinstance(n, ast.Name) and n.id == interceptor.name for n in ast.walk(server_assignment))
    methods = [n for cls in serve.body if isinstance(cls, ast.ClassDef) for n in cls.body
               if isinstance(n, ast.AsyncFunctionDef) and n.name in {"CreateVideo", "RenderGameReplay"}]

    async def exercise():
        async def authorized(*_args):
            return {"gameId": "test"}

        fail = Mock(side_effect=SearchTimeBudgetExceeded("censored live policy"))
        namespace = {
            "grpc": grpc, "json": json, "SearchTimeBudgetExceeded": SearchTimeBudgetExceeded,
            "_require_trusted": authorized, "_require_private_game": authorized,
            "VideoJobRequest": SimpleNamespace(model_validate=fail), "HistoricalReplayRenderRequest": fail,
            "LegacyVideoCompatibilityError": type("LegacyVideoCompatibilityError", (ValueError,), {}),
        }
        exec(compile(ast.Module(body=[interceptor, *methods], type_ignores=[]),
                     str(source), "exec"), namespace)
        server = grpc.aio.server(interceptors=[namespace[interceptor.name]()])

        async def handle(value, context):
            if value == b"ok":
                return b"ok"
            method = value.decode()
            if method == "direct":
                raise SearchTimeBudgetExceeded("censored live policy")
            request = SimpleNamespace(
                canonical_json=b"{}", identifier="test", decision_id="", renderer_plugin="", theme="",
                width=0, height=0, fps=0, crf=0, audit_samples=0, rerun_audit=False, maximum_audit_decisions=0,
            )
            return await namespace[method](None, request, context)

        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "test.Censor", {"Read": grpc.unary_unary_rpc_method_handler(handle)},
        ),))
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                read = channel.unary_unary("/test.Censor/Read")
                assert await read(b"ok", timeout=5) == b"ok"
                for method in ("direct", "CreateVideo", "RenderGameReplay"):
                    with pytest.raises(grpc.aio.AioRpcError) as refused:
                        await read(method.encode(), timeout=5)
                    assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                    assert json.loads(refused.value.details())["reason"] == "time_censored"
        finally:
            await server.stop(0)
            # Break the exec namespace cycle before asyncio closes the gRPC loop.
            namespace.clear()

    asyncio.run(exercise())


@pytest.mark.parametrize("source", ["consensus", "hitl"])
@pytest.mark.parametrize("censored", [False, True])
def test_hashed_dataset_preserves_and_flags_policy_censorship(root, rules, tmp_path, monkeypatch, source, censored):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=905, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(state, forced_contract=Contract.SUN, forced_buyer=Seat.SOUTH)
    engine.prepare_projects(state)
    legal = tuple(engine.legal_moves(state))
    completion = "time_censored" if censored else "heuristic_complete"
    metadata = {
        "analysisSolverCompletion": completion, "analysisCompletionStatuses": [completion],
        "source": "bot", "observerSafe": True, "privateHands": "must never enter the policy trace",
    }
    store = LearningStore(tmp_path / "learning.sqlite3")
    learning = LearningService(root, tmp_path / "learning", store=store)
    decision = store.record_decision(
        game_id="dataset-test", game_version=1, mode="full", state=state,
        actor=state.current_player, decision_kind="play", legal_actions=legal,
        selected_action=legal[0], decision_metadata=metadata,
    )
    reviewed = {
        **decision,
        "structuredReviews": [{"preferredAction": legal[0].to_dict(), "confidence": 0.99, "createdAt": "test"}],
        "reviewConsensus": {"trainingEligible": True, "preferredAction": legal[0].to_dict(), "supportShare": 1.0},
    }
    monkeypatch.setattr(store, "learning_queue", lambda **_kwargs: [{"decision": reviewed}] if source == "consensus" else [])
    if source == "hitl":
        learning.hitl_store = SimpleNamespace(
            list_candidates=lambda **_kwargs: [{"candidateId": "candidate", "caseId": "case", "preferredAction": legal[0].to_dict()}],
            get_case=lambda _id: {"caseId": "case", "decisionId": decision["decisionId"], "caseHash": "a" * 64},
        )
    dataset = learning.build_dataset(minimum_confidence=0.5)

    body = Path(dataset["examples"]).read_bytes()
    row, = [json.loads(line) for line in body.splitlines()]
    assert row["policyTrace"]["analysisSolverCompletion"] == completion
    assert row["policyTrace"]["analysisCompletionStatuses"] == [completion]
    assert "privateHands" not in row["policyTrace"]
    assert hashlib.sha256(body).hexdigest() == dataset["sha256"]
    assert dataset["manifestData"]["metadata"].get("timeCensoredExamples", 0) == int(censored)
    if not censored:
        assert "timeCensoredExamples" not in dataset["manifestData"]["metadata"]
