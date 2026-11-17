from __future__ import annotations

from copy import deepcopy
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from baloot_platform.service import api
from baloot_platform.service.api import (
    ResearchDomainCapabilitiesResponse,
    ResearchPlayCompilerVerificationResponse,
    ResearchResultRenderJobResponse,
    ResearchResultRenderListResponse,
    ResearchRulePrimitiveListResponse,
    ResearchRunResponse,
    ResearchSemanticDisagreementListResponse,
    ResearchSequentialSearchReceiptListResponse,
)
from baloot_platform.service.routes.configuration import (
    ConfigurationBundleListResponse,
)
from baloot_platform.storage.postgres import PostgresJobStore


HASH_A = "a" * 64
HASH_B = "b" * 64
RESULT_CODE = "LAF-ABCDEFGHIJ"
RUN_CODE = "RUN-ABCDEFGHIJKL"


def _v1_disagreement() -> dict:
    return {
        "schemaVersion": 1,
        "disagreementId": "DSG-ABCDEFGHIJKL",
        "runCode": RUN_CODE,
        "candidateHash": HASH_A,
        "category": "legality_mismatch",
        "solverEvidence": {"legal": True},
        "engineEvidence": {"legal": False},
        "status": "open",
        "sha256": HASH_B,
        "effectiveCategory": "historical_record_requires_readjudication",
        "semanticDefect": False,
    }


def _v2_disagreement() -> dict:
    return {
        "schemaVersion": 2,
        "disagreementId": "DSG-MNOPQRSTUVWX",
        "runCode": RUN_CODE,
        "candidateHash": HASH_B,
        "category": "compiler_parity_defect",
        "semanticDefect": True,
        "solverEvidence": {"status": "feasible"},
        "runtimeEvidence": {"status": "rejected"},
        "status": "open",
        "sha256": HASH_A,
        "effectiveCategory": "compiler_parity_defect",
    }


def _render_job(*, lease_until: float | None = None) -> dict:
    return {
        "jobId": "render-job-1",
        "tenantId": "default",
        "kind": "render_batch_v1",
        "status": "running" if lease_until is not None else "queued",
        "priority": 100,
        "progress": 0.25,
        "request": {"schemaVersion": 1, "items": []},
        "result": None,
        "error": None,
        "attempts": 1,
        "maxAttempts": 3,
        "leaseOwner": "worker-1" if lease_until is not None else None,
        "leaseUntil": lease_until,
        "cancelRequested": False,
        "createdAt": "2026-08-09T10:00:00+00:00",
        "updatedAt": "2026-08-09T10:01:00+00:00",
    }


def _verification() -> dict:
    return {
        "schemaVersion": 2,
        "resultCode": RESULT_CODE,
        "outcome": "matched",
        "solverStatus": "optimal",
        "projectSolverStatus": "optimal",
        "projectPlanHash": HASH_A,
        "projectPoints": [20, 0],
        "balootPoints": [0, 0],
        "projectRevealCount": 1,
        "playPlanHash": HASH_B,
        "evidenceHash": HASH_A,
        "evidence": {
            "uri": f"cas://sha256/{HASH_B}",
            "sha256": HASH_B,
            "bytes": 512,
            "mediaType": (
                "application/vnd.balootx.cp-sat-round-parity-v2+json"
            ),
            "name": f"{HASH_A}.json",
        },
    }


def _research_run(*, status: str = "queued") -> dict:
    return {
        "runCode": RUN_CODE,
        "runHash": HASH_A,
        "recipeCode": "RCP-APICONTRACT",
        "recipeHash": HASH_B,
        "generationLane": "ortools",
        "status": status,
        "progress": 0.0,
        "events": [],
        "createdAt": "2026-08-09T10:00:00+00:00",
    }


def test_legacy_run_read_shape_remains_readable_without_inventing_a_lane() -> None:
    legacy = _research_run()
    legacy.pop("generationLane")
    legacy["optimalityCutCount"] = 2
    legacy["error"] = {"type": "RuntimeError", "message": "legacy", "attemptCount": 1}
    parsed = ResearchRunResponse.model_validate(legacy)
    assert parsed.generationLane is None
    assert "optimalityCutCount" not in parsed.model_dump()
    assert parsed.error is not None and parsed.error.message == "legacy"


def _sequential_receipt_summary() -> dict:
    return {
        "admittedSeedCount": 1,
        "exactRejectionCount": 2,
        "noGoodAppliedCount": 2,
        "noGoodUnappliedCount": 0,
        "continuationSolveCount": 3,
        "continuationStatusDistribution": {
            "optimal": 2,
            "infeasible": 1,
        },
        "stopReasonDistribution": {
            "candidates_exhausted": 0,
            "maximum_solutions_reached": 0,
            "qualified": 1,
            "incomplete": 0,
            "cancelled": 0,
            "fatal_quarantine": 0,
        },
    }


def _sequential_receipt() -> dict:
    receipt_bundle = {
        "schemaVersion": 1,
        "kind": "research_sequential_candidate_search_receipt_v1",
        "rootSeed": 37,
        "emittedCandidateCount": 3,
        "evaluatedCandidateCount": 3,
        "rejectedCandidateCount": 2,
        "evaluationReceipts": [{"candidateOrdinal": 0}],
        "rejectionFeedbackRows": [{"rejectionOrdinal": 0}],
        "terminalContinuationReceipt": None,
        "qualifiedCandidateHash": HASH_A,
        "stopReason": "qualified",
        "failureCodes": [],
        "sha256": HASH_B,
    }
    return {
        "schemaVersion": 1,
        "occurrenceHash": HASH_A,
        "receiptHash": HASH_B,
        "runCode": RUN_CODE,
        "shardId": "RSH-ABCDEFGHIJKLMNOP",
        "attemptCount": 2,
        "fencingToken": 3,
        "seed": 37,
        "admitted": True,
        "createdAt": "2026-08-09T10:02:00.000000+00:00",
        "summary": _sequential_receipt_summary(),
        "receiptBundle": receipt_bundle,
    }


def test_live_catalog_reads_validate_without_changing_optional_field_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BALOOT_API_KEY", "research-read-contract")
    headers = {
        "x-api-key": "research-read-contract",
        "x-baloot-subject-id": "TEST-RESEARCH-READ",
    }
    with TestClient(api.app) as client:
        capabilities = client.get(
            "/api/v2/domain-capabilities", headers=headers
        )
        payload_capabilities = client.get(
            "/api/v2/domain-capabilities",
            params={"includePayloads": True},
            headers=headers,
        )
        primitives = client.get("/api/v2/rule-primitives", headers=headers)
        bundles = client.get(
            "/api/v2/configuration/bundles",
            params={"status": "published", "limit": 2},
            headers=headers,
        )

    assert capabilities.status_code == 200, capabilities.text
    capabilities_value = capabilities.json()
    assert not {"rules", "conventions", "localization"}.intersection(
        capabilities_value
    )
    ResearchDomainCapabilitiesResponse.model_validate(capabilities_value)

    assert payload_capabilities.status_code == 200, payload_capabilities.text
    payload_value = payload_capabilities.json()
    assert {"rules", "conventions", "localization"}.issubset(payload_value)
    ResearchDomainCapabilitiesResponse.model_validate(payload_value)

    assert primitives.status_code == 200, primitives.text
    primitive_value = primitives.json()
    assert "configurationSelection" not in primitive_value
    ResearchRulePrimitiveListResponse.model_validate(primitive_value)

    assert bundles.status_code == 200, bundles.text
    ConfigurationBundleListResponse.model_validate(bundles.json())

    invalid = {**capabilities_value, "unknown": True}
    with pytest.raises(ValidationError):
        ResearchDomainCapabilitiesResponse.model_validate(invalid)


def test_semantic_disagreement_union_is_closed_and_preserves_v1_v2_reads() -> None:
    payload = {"items": [_v1_disagreement(), _v2_disagreement()]}
    model = ResearchSemanticDisagreementListResponse.model_validate(payload)
    assert model.model_dump(mode="json") == payload

    invalid = deepcopy(payload)
    invalid["items"][1]["unexpected"] = True
    with pytest.raises(ValidationError):
        ResearchSemanticDisagreementListResponse.model_validate(invalid)


class _ResearchReadStub:
    def __init__(self) -> None:
        self.disagreement_filters: dict | None = None
        self.receipt_calls: list[tuple[str, int, int]] = []
        self.verified: list[str] = []

    def list_disagreements(self, **filters):
        self.disagreement_filters = filters
        return [_v1_disagreement(), _v2_disagreement()]

    def get_result(self, code: str):
        return {"resultCode": code} if code == RESULT_CODE else None

    def get_run(self, code: str):
        return {"runCode": code} if code == RUN_CODE else None

    def list_sequential_search_receipts(
        self,
        run_code: str,
        *,
        offset: int,
        limit: int,
    ):
        self.receipt_calls.append((run_code, offset, limit))
        return {
            "items": [_sequential_receipt()],
            "total": 10,
            "offset": offset,
            "limit": limit,
            "nextOffset": offset + 1,
            "hasMore": True,
        }

    def verify_play_compiler(self, code: str):
        self.verified.append(code)
        if code != RESULT_CODE:
            raise KeyError(code)
        return _verification()


class _RenderReadStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def list_for_source(self, *, source_kind: str, source_code: str):
        self.calls.append((source_kind, source_code))
        return [_render_job()]


def test_http_read_envelopes_are_exact_and_forward_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research = _ResearchReadStub()
    renders = _RenderReadStub()
    monkeypatch.setattr(api, "RESEARCH", research)
    monkeypatch.setattr(api, "RENDER_BATCHES", renders)

    with TestClient(api.app) as client:
        disagreements = client.get(
            "/api/v2/semantic-disagreements",
            params={"runCode": RUN_CODE, "status": "open", "limit": 2},
        )
        render_jobs = client.get(
            f"/api/v2/research-results/{RESULT_CODE}/renders"
        )
        verification = client.post(
            f"/api/v2/research-results/{RESULT_CODE}:verify-play-compiler"
        )
        receipts = client.get(
            f"/api/v2/research-runs/{RUN_CODE}/sequential-search-receipts",
            params={"offset": 4, "limit": 7},
        )
        missing_receipts = client.get(
            "/api/v2/research-runs/RUN-ZZZZZZZZZZZZ/sequential-search-receipts"
        )
        invalid_receipt_page = client.get(
            f"/api/v2/research-runs/{RUN_CODE}/sequential-search-receipts",
            params={"offset": -1, "limit": 0},
        )
        missing_renders = client.get(
            "/api/v2/research-results/LAF-ZZZZZZZZZZ/renders"
        )

    assert disagreements.status_code == 200, disagreements.text
    assert disagreements.json() == {
        "items": [_v1_disagreement(), _v2_disagreement()]
    }
    assert research.disagreement_filters == {
        "run_code": RUN_CODE,
        "status": "open",
        "limit": 2,
    }

    assert render_jobs.status_code == 200, render_jobs.text
    expected_renders = {"resultCode": RESULT_CODE, "items": [_render_job()]}
    assert render_jobs.json() == expected_renders
    ResearchResultRenderListResponse.model_validate(expected_renders)
    assert renders.calls == [("research_scenario_result", RESULT_CODE)]
    assert missing_renders.status_code == 404

    assert verification.status_code == 200, verification.text
    assert verification.json() == _verification()
    ResearchPlayCompilerVerificationResponse.model_validate(
        verification.json()
    )
    assert research.verified == [RESULT_CODE]

    assert receipts.status_code == 200, receipts.text
    expected_receipts = {
        "items": [_sequential_receipt()],
        "total": 10,
        "offset": 4,
        "limit": 7,
        "nextOffset": 5,
        "hasMore": True,
    }
    assert receipts.json() == expected_receipts
    ResearchSequentialSearchReceiptListResponse.model_validate(expected_receipts)
    assert receipts.json()["items"][0]["receiptBundle"] == (
        _sequential_receipt()["receiptBundle"]
    )
    assert research.receipt_calls == [(RUN_CODE, 4, 7)]
    assert missing_receipts.status_code == 404
    assert invalid_receipt_page.status_code == 422


@pytest.mark.parametrize(
    ("error", "status_code"),
    ((KeyError("missing"), 404), (ValueError("invalid"), 409),
     (RuntimeError("unavailable"), 503)),
)
def test_play_compiler_verification_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    class _FailingResearch:
        def verify_play_compiler(self, code: str):
            raise error

    monkeypatch.setattr(api, "RESEARCH", _FailingResearch())
    with TestClient(api.app) as client:
        response = client.post(
            f"/api/v2/research-results/{RESULT_CODE}:verify-play-compiler"
        )
    assert response.status_code == status_code


@pytest.fixture
def private_family_progress():
    return {
        "type": "progress", "sequence": 17, "currentStage": "family_proposal",
        "at": "2026-08-09T10:00:00+00:00", "resultCount": 0,
        "screenedSeed": 6, "screenedDealer": "south",
        "auctionQualification": {"status": "complete", "contract": "sun", "allPassed": False},
        "incompleteCompilation": {"privateEvidence": "PRIVATE-OPPONENT-CARDS"},
        "futurePlays": [{"actor": "east", "card": "SA"}],
        "principalVariation": ["SA"], "roundScore": [64, 6],
    }


@pytest.mark.real_clock
def test_research_run_cancel_and_event_routes_expose_real_http_contracts(
    monkeypatch: pytest.MonkeyPatch,
    private_family_progress,
) -> None:
    class _RunStub:
        def __init__(self) -> None:
            self.run = {**_research_run(), "events": [private_family_progress]}

        def cancel(self, run_code: str):
            if run_code == "RUN-ZZZZZZZZZZZZ":
                raise KeyError(run_code)
            if run_code == "RUN-TERMINAL0001":
                raise ValueError(
                    "research run cannot be cancelled from its current state"
                )
            self.run = {**self.run, "status": "cancelled"}
            return self.run

        def get_run(self, run_code: str):
            return self.run if run_code == RUN_CODE else None

        def events(self, run_code: str):
            if run_code != RUN_CODE:
                raise KeyError(run_code)
            return [private_family_progress,
                    {"type": "cancelled", "at": "2026-08-09T10:01:00+00:00"}]

    monkeypatch.setattr(api, "RESEARCH", _RunStub())
    with TestClient(api.app) as client:
        cancelled = client.post(f"/api/v2/research-runs/{RUN_CODE}:cancel")
        missing_cancel = client.post(
            "/api/v2/research-runs/RUN-ZZZZZZZZZZZZ:cancel"
        )
        terminal_cancel = client.post(
            "/api/v2/research-runs/RUN-TERMINAL0001:cancel"
        )
        events = client.get(f"/api/v2/research-runs/{RUN_CODE}/events")
        missing_events = client.get(
            "/api/v2/research-runs/RUN-ZZZZZZZZZZZZ/events"
        )

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["events"] == [{key: private_family_progress[key] for key in (
        "type", "sequence", "currentStage", "at", "resultCount")}]
    assert missing_cancel.status_code == 404
    assert missing_cancel.json() == {"detail": "research run not found"}
    assert terminal_cancel.status_code == 409
    assert terminal_cancel.json() == {
        "detail": "research run cannot be cancelled from its current state"
    }
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert events.headers["cache-control"] == "no-cache"
    assert events.headers["x-accel-buffering"] == "no"
    assert "event: cancelled" in events.text
    for response in (cancelled, events):
        for private_key in ("screenedSeed", "screenedDealer", "auctionQualification",
                            "PRIVATE-OPPONENT-CARDS", "futurePlays", "principalVariation", "roundScore"):
            assert private_key not in response.text
    assert missing_events.status_code == 404
    assert missing_events.json() == {"detail": "research run not found"}


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_research_event_stream_uses_operator_projection(monkeypatch, private_family_progress, status):
    class ActiveRun:
        streamed = False

        def get_run(self, code):
            assert code == RUN_CODE
            return {**_research_run(), "status": "completed" if self.streamed else status}

        def events(self, code):
            assert code == RUN_CODE
            self.streamed = True
            return [private_family_progress]

    monkeypatch.setattr(api, "RESEARCH", ActiveRun())
    with TestClient(api.app) as client:
        response = client.get(f"/api/v2/research-runs/{RUN_CODE}/events")
    assert response.status_code == 200
    assert "id: 17\n" in response.text
    event = json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith("data: ")))
    assert event == {key: private_family_progress[key] for key in (
        "type", "sequence", "currentStage", "at", "resultCount")}


@pytest.mark.real_clock
def test_cancel_and_event_http_projection_preserves_private_sqlite_evidence(
    monkeypatch, tmp_path, private_family_progress,
):
    import threading
    from baloot_platform.research.service import ResearchFoundryServiceV1
    from baloot_platform.research.store import SQLiteResearchStoreV1

    store = SQLiteResearchStoreV1(tmp_path / "private-events.sqlite3")
    try:
        run = {**_research_run(status="running"), "events": [private_family_progress]}
        store.put_recipe({"code": run["recipeCode"], "sha256": run["recipeHash"], "status": "published"})
        store.create_run(run)
        research = object.__new__(ResearchFoundryServiceV1)
        research.store, research._lock, research._cancelled = store, threading.Lock(), set()
        monkeypatch.setattr(api, "RESEARCH", research)
        with TestClient(api.app) as client:
            cancelled = client.post(f"/api/v2/research-runs/{RUN_CODE}:cancel")
            streamed = client.get(f"/api/v2/research-runs/{RUN_CODE}/events")
            read = client.get(f"/api/v2/research-runs/{RUN_CODE}")
        assert cancelled.status_code == streamed.status_code == read.status_code == 200
        assert cancelled.json() == read.json()
        for response in (cancelled, streamed, read):
            for private_key in ("screenedSeed", "screenedDealer", "auctionQualification",
                                "PRIVATE-OPPONENT-CARDS", "futurePlays", "principalVariation", "roundScore"):
                assert private_key not in response.text
        assert store.get_run(RUN_CODE)["status"] == "cancelled"
        assert research.events(RUN_CODE)[0] == private_family_progress
        assert store.get_run(RUN_CODE)["events"][0] == private_family_progress
        assert RUN_CODE in research._cancelled
    finally:
        store.close()


def test_research_event_stream_uses_sequence_after_the_rolling_window_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RollingRunStub:
        def __init__(self) -> None:
            self.event_reads = 0
            self.run_reads = 0

        def get_run(self, run_code: str):
            assert run_code == RUN_CODE
            self.run_reads += 1
            return {
                **_research_run(),
                "status": "completed" if self.run_reads >= 3 else "running",
            }

        def events(self, run_code: str):
            assert run_code == RUN_CODE
            self.event_reads += 1
            start = 1 if self.event_reads == 1 else 2
            return [
                {
                    "type": "completed" if sequence == 201 else "progress",
                    "sequence": sequence,
                    "at": f"2026-08-30T00:00:{sequence % 60:02d}+00:00",
                }
                for sequence in range(start, start + 200)
            ]

    stub = _RollingRunStub()
    monkeypatch.setattr(api, "RESEARCH", stub)
    with TestClient(api.app) as client:
        response = client.get(f"/api/v2/research-runs/{RUN_CODE}/events")

    assert response.status_code == 200
    assert response.text.count("event: progress") == 200
    assert response.text.count("event: completed") == 1
    assert "id: 201\n" in response.text
    assert stub.event_reads == 2


def test_research_event_stream_resumes_from_last_event_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReconnectRunStub:
        def get_run(self, run_code: str):
            assert run_code == RUN_CODE
            return {**_research_run(), "status": "completed"}

        def events(self, run_code: str):
            assert run_code == RUN_CODE
            return [
                {
                    "type": "completed" if sequence == 201 else "progress",
                    "sequence": sequence,
                    "at": "2026-08-30T00:00:00+00:00",
                }
                for sequence in range(2, 202)
            ]

    monkeypatch.setattr(api, "RESEARCH", _ReconnectRunStub())
    with TestClient(api.app) as client:
        response = client.get(
            f"/api/v2/research-runs/{RUN_CODE}/events",
            headers={"Last-Event-ID": "200"},
        )

    assert response.status_code == 200
    assert response.text.count("event: progress") == 0
    assert response.text.count("event: completed") == 1
    assert "id: 201\n" in response.text


class _Cursor:
    def __init__(self, row: tuple) -> None:
        self.row = row

    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, row: tuple) -> None:
        self.row = row

    def execute(self, _query: str, _parameters: tuple):
        return _Cursor(self.row)


class _ConnectionContext:
    def __init__(self, row: tuple) -> None:
        self.connection = _Connection(row)

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False


class _Pool:
    def __init__(self, row: tuple) -> None:
        self.row = row

    def connection(self):
        return _ConnectionContext(self.row)


@pytest.mark.real_clock
def test_postgres_job_lease_deadline_matches_sqlite_epoch_seconds() -> None:
    lease_at = datetime(2026, 8, 9, 10, 5, tzinfo=timezone.utc)
    created_at = datetime(2026, 8, 9, 10, 0, tzinfo=timezone.utc)
    row = (
        "render-job-1",
        "default",
        "render_batch_v1",
        "running",
        100,
        0.25,
        {"schemaVersion": 1, "items": []},
        None,
        None,
        1,
        3,
        "worker-1",
        lease_at,
        False,
        created_at,
        created_at,
    )
    store = object.__new__(PostgresJobStore)
    store.pool = _Pool(row)

    value = store.get("render-job-1")
    assert value is not None
    assert value["leaseUntil"] == lease_at.timestamp()
    assert isinstance(value["leaseUntil"], float)
    ResearchResultRenderJobResponse.model_validate(value)


def test_openapi_read_contracts_are_named_closed_and_discriminated() -> None:
    expected = {
        ("/api/v2/domain-capabilities", "get"):
            "ResearchDomainCapabilitiesResponse",
        ("/api/v2/configuration/bundles", "get"):
            "ConfigurationBundleListResponse",
        ("/api/v2/rule-primitives", "get"):
            "ResearchRulePrimitiveListResponse",
        ("/api/v2/semantic-disagreements", "get"):
            "ResearchSemanticDisagreementListResponse",
        ("/api/v2/research-results/{code}/renders", "get"):
            "ResearchResultRenderListResponse",
        ("/api/v2/research-results/{code}:verify-play-compiler", "post"):
            "ResearchPlayCompilerVerificationResponse",
        ("/api/v2/research-runs/{run_code}/sequential-search-receipts", "get"):
            "ResearchSequentialSearchReceiptListResponse",
    }
    openapi = api.app.openapi()
    for (path, method), model_name in expected.items():
        schema = openapi["paths"][path][method]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]
        assert schema == {"$ref": f"#/components/schemas/{model_name}"}
        assert openapi["components"]["schemas"][model_name][
            "additionalProperties"
        ] is False

    for model_name in (
        "ConfigurationBundleRevisionResponse",
        "ResearchCapabilityResponse",
        "ResearchConfigurationSelectionReceiptResponse",
        "ResearchPlayCompilerVerificationArtifactResponse",
        "ResearchResultRenderJobResponse",
        "ResearchRulePrimitiveResponse",
        "ResearchSemanticDisagreementV1Response",
        "ResearchSemanticDisagreementV2Response",
        "ResearchSequentialContinuationStatusDistributionResponse",
        "ResearchSequentialSearchReceiptResponse",
        "ResearchSequentialSearchReceiptSummaryResponse",
        "ResearchSequentialStopReasonDistributionResponse",
    ):
        assert openapi["components"]["schemas"][model_name][
            "additionalProperties"
        ] is False

    render_required = set(
        openapi["components"]["schemas"]["ResearchResultRenderJobResponse"][
            "required"
        ]
    )
    assert {"result", "error", "leaseOwner", "leaseUntil"}.issubset(
        render_required
    )

    disagreement_items = openapi["components"]["schemas"][
        "ResearchSemanticDisagreementListResponse"
    ]["properties"]["items"]["items"]
    assert disagreement_items["discriminator"] == {
        "propertyName": "schemaVersion",
        "mapping": {
            "1": "#/components/schemas/ResearchSemanticDisagreementV1Response",
            "2": "#/components/schemas/ResearchSemanticDisagreementV2Response",
        },
    }

    receipt_operation = openapi["paths"][
        "/api/v2/research-runs/{run_code}/sequential-search-receipts"
    ]["get"]
    receipt_parameters = {
        item["name"]: item for item in receipt_operation["parameters"]
    }
    assert receipt_parameters["offset"]["schema"] == {
        "type": "integer",
        "minimum": 0,
        "default": 0,
        "title": "Offset",
    }
    assert receipt_parameters["limit"]["schema"] == {
        "type": "integer",
        "maximum": 500,
        "minimum": 1,
        "default": 100,
        "title": "Limit",
    }

    error_schema = {
        "$ref": "#/components/schemas/ResearchMutationErrorResponse"
    }
    cancel = openapi["paths"][
        "/api/v2/research-runs/{run_code}:cancel"
    ]["post"]
    assert cancel["responses"]["404"]["content"]["application/json"][
        "schema"
    ] == error_schema
    assert cancel["responses"]["409"]["content"]["application/json"][
        "schema"
    ] == error_schema

    events = openapi["paths"][
        "/api/v2/research-runs/{run_code}/events"
    ]["get"]
    assert events["responses"]["200"]["content"] == {
        "text/event-stream": {"schema": {"type": "string"}}
    }
    assert events["responses"]["404"]["content"]["application/json"][
        "schema"
    ] == error_schema

    verification = openapi["paths"][
        "/api/v2/research-results/{code}:verify-play-compiler"
    ]["post"]
    for status in ("404", "409", "503"):
        assert verification["responses"][status]["content"][
            "application/json"
        ]["schema"] == error_schema
    assert (
        "/api/v2/research-runs/{run_code}/sequential-search-receipts/{seed}"
        not in openapi["paths"]
    )

    def walk_routes(routes):
        for route in routes:
            yield route
            nested = getattr(route, "routes", None)
            if nested is None:
                original_router = getattr(route, "original_router", None)
                nested = getattr(original_router, "routes", None)
            if nested:
                yield from walk_routes(nested)

    routes = list(walk_routes(api.app.routes))
    for key, model_name in expected.items():
        path, method = key
        route = next(
            route
            for route in routes
            if getattr(route, "path", None) == path
            and method.upper() in getattr(route, "methods", set())
        )
        assert route.response_model.__name__ == model_name
