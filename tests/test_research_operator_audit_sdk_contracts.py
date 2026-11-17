from __future__ import annotations

import json
from pathlib import Path

import httpx

from sdk.python.client import BalootClient


ROOT = Path(__file__).resolve().parents[1]


def test_python_sdk_retrieves_the_typed_operator_audit() -> None:
    requests: list[httpx.Request] = []
    payload = {
        "schemaVersion": 1,
        "resultCode": "LAF-ABCDEF1234",
        "researchAuthority": {
            "schemaVersion": 1,
            "backendId": "ortools_cp_sat_v3",
            "backendVersion": "3.0.0",
            "qualificationAuthority": "cp_sat",
            "solverVersion": "9.14",
            "bundleRevisionId": "CBR-SDK-V4",
            "bundleHash": "a" * 64,
            "domainIrHashes": {"rules": "b" * 64},
            "loweringManifestHash": "c" * 64,
            "constructionModelHash": "d" * 64,
            "proofModelHash": "e" * 64,
            "graphHash": "f" * 64,
            "traceHash": "1" * 64,
            "proofOutcomeHash": "2" * 64,
            "sha256": "3" * 64,
        },
        "runtimeCompatibility": {
            "schemaVersion": 1,
            "status": "passed",
            "engineVersion": "21.2.0",
            "bundleHash": "a" * 64,
            "traceHash": "1" * 64,
            "evidenceHash": "4" * 64,
            "failureCodes": [],
            "sha256": "5" * 64,
        },
        "executionReceipt": {
            "schemaVersion": 1,
            "semanticResultHash": "6" * 64,
            "stageTimings": {"proof": 0.25},
            "cpuSeconds": 0.2,
            "peakMemoryBytes": 1024,
            "conflicts": 3,
            "branches": 4,
            "cacheBytes": 512,
            "completedVariants": 1,
            "machine": {"platform": "test"},
            "sha256": "7" * 64,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    client = BalootClient(base_url="http://sdk.test")
    client.client.close()
    client.client = httpx.Client(
        base_url="http://sdk.test", transport=httpx.MockTransport(handler)
    )
    try:
        assert client.research_result_operator_audit("LAF-ABCDEF1234") == payload
    finally:
        client.close()

    assert requests[0].url.path == (
        "/api/v2/research-results/LAF-ABCDEF1234/operator-audit"
    )


def test_handwritten_sdks_expose_strict_v4_operator_audit_models() -> None:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(
        encoding="utf-8"
    )
    go_source = (ROOT / "sdk/go/client.go").read_text(encoding="utf-8")

    for model in (
        "ResearchAuthorityV1",
        "RuntimeCompatibilityV1",
        "ExecutionReceiptV1",
        "ResearchOperatorAuditV1Response",
    ):
        assert f"class {model}(TypedDict):" in python_source
        assert f"export interface {model}" in typescript_source
        assert f"type {model} struct" in go_source

    assert "def research_result_operator_audit(" in python_source
    assert ") -> ResearchOperatorAuditV1Response:" in python_source
    assert "researchResultOperatorAudit(code:string):Promise<ResearchOperatorAuditV1Response>" in typescript_source
    assert ") ResearchResultOperatorAudit(" in go_source


def test_openapi_operator_audit_contract_matches_the_handwritten_sdk_models() -> None:
    document = json.loads((ROOT / "api/openapi.json").read_text(encoding="utf-8"))
    operation = document["paths"][
        "/api/v2/research-results/{result_code}/operator-audit"
    ]["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ResearchOperatorAuditV1Response"
    }
    schemas = document["components"]["schemas"]
    response = schemas["ResearchOperatorAuditV1Response"]
    assert response["additionalProperties"] is False
    assert set(response["required"]) == {
        "resultCode",
        "researchAuthority",
        "runtimeCompatibility",
        "executionReceipt",
    }
    for name in (
        "ResearchAuthorityV1Response",
        "RuntimeCompatibilityV1Response",
        "ExecutionReceiptV1Response",
    ):
        assert schemas[name]["additionalProperties"] is False
    assert schemas["ResearchAuthorityV1Response"]["properties"]["backendId"][
        "enum"
    ] == ["ortools_cp_sat_v2", "ortools_cp_sat_v3"]


def test_openapi_exposes_the_closed_playing_quality_audit() -> None:
    document = json.loads((ROOT / "api/openapi.json").read_text(encoding="utf-8"))
    operation = document["paths"][
        "/api/v2/research-results/{result_code}/quality-audit"
    ]["get"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ResearchPlayingQualityAuditV1Response"
    }
    schemas = document["components"]["schemas"]
    assert schemas["ResearchPlayingQualityAuditV1Response"]["additionalProperties"] is False
    assert schemas["ResearchPlayingQualityDecisionV1Response"]["additionalProperties"] is False
