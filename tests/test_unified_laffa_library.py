from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

import pytest

from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.store import SQLiteResearchStoreV1
from baloot_platform.service.laffa_library import LaffaLibraryServiceV1
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _digest(character: str) -> str:
    return character * 64


class _Learning:
    def list_games(self, *, query: str = "", offset: int = 0, limit: int = 100):
        return [{
            "gameHash": _digest("a"),
            "serial": "LAF-ARCHIVE",
            "contract": "hokum",
            "buyer": "south",
            "createdAt": "2026-08-01T00:00:00Z",
        }][offset : offset + limit]

    def get_game(self, identifier: str):
        return {
            "gameHash": identifier,
            "state": {"contract": "hokum", "buyer": "south", "dealer": "east"},
            "verification": {"valid": True},
        }

    def verify_game(self, _identifier: str):
        return {"valid": True}

    def list_video_records(self, **_filters):
        return [{"videoId": "VIDEO-1", "gameHash": _digest("a")}]


class _Catalogs:
    def list(self, **_filters):
        return {
            "items": [{"catalogHash": _digest("c"), "createdAt": "2026-08-03T00:00:00Z"}],
            "nextCursor": None,
        }

    def get(self, catalog_hash: str):
        if catalog_hash == _digest("e"):
            return {
                "record": {"catalogHash": catalog_hash},
                "document": {
                    "traceReferences": [{
                        "ownerKind": "complete_trace_v1",
                        "ownerHash": _digest("b"),
                        "uri": "cas://sha256/" + _digest("f"),
                        "sha256": _digest("f"),
                        "bytes": 100,
                        "mediaType": "application/vnd.balootx.complete-trace-v1+json",
                    }],
                },
            }
        assert catalog_hash == _digest("c")
        return {
            "record": {"catalogHash": catalog_hash},
            "document": {
                "preparedItem": {
                    "traceHash": _digest("b"),
                    "shortCode": "LAF-QUALIFIED",
                    "truthLabel": "exact_optimal",
                    "confidence": 0.98,
                    "interestingProfiles": ["only_winning_move"],
                    "laffaSelectionScore": {"score": 94.0},
                    "bundleArtifact": {"uri": "cas://sha256/" + _digest("d")},
                }
            },
        }

    def list_sakkas(self, **_filters):
        return {
            "items": [{
                "catalogHash": _digest("e"),
                "kind": "natural_sakka_v1",
                "createdAt": "2026-08-04T00:00:00Z",
            }],
            "nextCursor": None,
        }

    def load_sakka(self, catalog_hash: str):
        assert catalog_hash == _digest("e")
        return SimpleNamespace(rounds=({"traceHash": _digest("b")},))


class _Renders:
    def list(self, *, limit: int):
        assert limit == 5_000
        return [{
            "jobId": "RENDER-1",
            "status": "queued",
            "request": {
                "items": [{
                    "source": {
                        "kind": "research_result",
                        "code": "LAF-RESEARCH",
                        "traceHash": _digest("b"),
                    }
                }]
            },
        }]


def _research(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    store.put_recipe({
        "code": "RCP-LIBRARY",
        "sha256": _digest("1"),
        "status": "published",
    })
    store.create_run({
        "runCode": "RUN-LIBRARY",
        "runHash": _digest("2"),
        "recipeCode": "RCP-LIBRARY",
        "recipeHash": _digest("1"),
        "status": "completed",
        "progress": 1.0,
    })
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    trace = json.dumps({
        "trustedFinalState": {
            "contract": "hokum",
            "buyer": "south",
            "dealer": "east",
        }
    }, sort_keys=True).encode("utf-8")
    reference = artifacts.put(
        trace,
        media_type="application/vnd.balootx.complete-trace-v1+json",
    )
    store.put_result({
        "resultCode": "LAF-RESEARCH",
        "resultHash": _digest("3"),
        "runCode": "RUN-LIBRARY",
        "traceHash": _digest("b"),
        "gameHash": _digest("a"),
        "truthLabel": "robust_best_known",
        "confidence": 0.91,
        "score": 88.0,
        "disposition": "qualified",
        "tacticalTags": ["small_card_big_gain"],
        "verifiedTacticalTags": ["only_winning_move"],
        "interestingSeats": ["south"],
        "libraryProjection": {
            "contract": "hokum",
            "projectKinds": ["four_hundred"],
            "actingSeats": ["south"],
        },
        "artifacts": {"completeTrace": reference},
        "createdAt": "2026-08-02T00:00:00Z",
    })
    return SimpleNamespace(
        store=store,
        list_results=store.list_results,
        artifact_store=artifacts,
    ), store


def test_sqlite_research_and_authoritative_sources_form_one_deduplicated_projection(tmp_path):
    research, store = _research(tmp_path)
    try:
        service = LaffaLibraryServiceV1(
            learning_store=_Learning(),
            research=research,
            foundry_catalogs=_Catalogs(),
            render_batches=_Renders(),
        )
        response = service.list(limit=10)
        assert response["resourceKind"] == "unified_laffa_library_v1"
        assert response["total"] == 1
        item = response["items"][0]
        assert item["humanCode"] == "LAF-QUALIFIED"
        assert item["sourceKinds"] == [
            "archived_game", "qualified_laffa", "research_result", "sakka_round"
        ]
        assert item["replayValid"] is True
        assert item["qualificationStatus"] == "qualified"
        assert item["truthLabel"] == "exact_optimal"
        assert item["challengeScore"] == 94.0
        assert item["tacticalTags"] == ["only_winning_move", "small_card_big_gain"]
        assert item["contract"] == "hokum" and item["project"] == "four_hundred"
        assert item["projects"] == ["four_hundred"] and item["seat"] == "south"
        assert item["buyer"] == "south" and item["dealer"] == "east"
        assert item["relatedSakkas"][0]["identity"] == _digest("e")
        assert item["renderCount"] == 2 and item["renderStatus"] == "queued"
        assert item["sourceLocator"]["uri"] == "/api/v2/foundry-catalogs/" + _digest("c")
        assert len(item["sourceReferences"]) == 4
        for identifier in (
            "LAF-QUALIFIED",
            "LAF-RESEARCH",
            _digest("a"),
            _digest("b"),
            _digest("c"),
        ):
            assert service.get(identifier) == item
        assert service.get("LAF-MISSING") is None
        resolved = service.resolve_sakka_round("LAF-" + _digest("b")[:10].upper())
        assert resolved is not None
        assert resolved["traceReference"]["ownerHash"] == _digest("b")

        filtered = service.list(
            source_kind="research_result",
            replay_valid=True,
            contract="hokum",
            project="four_hundred",
            tactic="only_winning_move",
            seat="south",
            render_status="queued",
            sort="score_desc",
            offset=0,
            limit=1,
        )
        assert filtered["total"] == 1 and filtered["nextOffset"] is None
        assert {item["sourceKind"]: item["count"] for item in filtered["sourceFacets"]} == {
            "archived_game": 1,
            "qualified_laffa": 1,
            "research_audit_trace": 0,
            "research_result": 1,
            "sakka_round": 1,
        }
    finally:
        store.close()


def test_scenario_v4_qualification_is_not_projected_as_challenge_quality(tmp_path):
    research, store = _research(tmp_path)
    try:
        store.put_result({
            "resultCode": "LAF-SCENARIO-V4",
            "resultHash": _digest("8"),
            "runCode": "RUN-LIBRARY",
            "traceHash": _digest("9"),
            "classification": "scenario_v4",
            "truthLabel": "exact",
            "confidence": 1.0,
            "score": 100.0,
            "disposition": "qualified",
            "createdAt": "2026-08-02T00:00:00Z",
        })
        service = LaffaLibraryServiceV1(
            learning_store=_Learning(), research=research,
            foundry_catalogs=_Catalogs(), render_batches=_Renders(),
        )
        item = service.get("LAF-SCENARIO-V4")
        assert item is not None
        assert item["challengeScore"] is None
        assert item["confidence"] is None
        assert item["truthLabel"] == "exact"
        assert item["qualificationStatus"] == "qualified"
        assert item["publicationEligible"] is False
        assert item["auditOnly"] is True
        assert "publication_eligibility_not_established" in item["qualificationFailures"]
        assert service.get("LAF-RESEARCH")["challengeScore"] == 94.0
    finally:
        store.close()


@pytest.mark.parametrize("source_kind", ["research_result", "research_scenario_result"])
def test_research_render_links_by_code_without_source_trace_hash(tmp_path, source_kind):
    research, store = _research(tmp_path)
    code = "LAF-SCENARIO-V4"
    jobs = [
        {
            "jobId": job_id,
            "status": status,
            "request": {"items": [{"source": {"kind": kind, "code": source_code}}]},
        }
        for job_id, status, kind, source_code in (
            ("RENDER-SCENARIO", "completed", source_kind, code),
            ("RENDER-FOREIGN", "running", source_kind, "LAF-FOREIGN"),
            ("RENDER-NONRESEARCH", "running", "laffa", code),
        )
    ]
    original_jobs = json.dumps(jobs, sort_keys=True)
    try:
        store.put_result({
            "resultCode": code,
            "resultHash": _digest("8"),
            "runCode": "RUN-LIBRARY",
            "traceHash": _digest("9"),
            "classification": "scenario_v4",
            "truthLabel": "exact",
            "confidence": 0.0,
            "score": 0.0,
            "disposition": "qualified",
        })
        service = LaffaLibraryServiceV1(
            learning_store=_Learning(), research=research, foundry_catalogs=None,
            render_batches=SimpleNamespace(list=lambda **_filters: jobs),
        )
        item = service.get(code)
        assert item is not None
        assert item["renderCount"] == 1 and item["renderStatus"] == "completed"
        assert item["sourceLocator"]["kind"] == "research_result"
        assert service.list(search=code, render_status="completed")["items"] == [item]
        assert service.list(search=code, render_status="not_prepared")["total"] == 0
        assert json.dumps(jobs, sort_keys=True) == original_jobs
    finally:
        store.close()


def test_application_openapi_types_the_unified_laffa_library_contract():
    from baloot_platform.service.api import app

    operation = app.openapi()["paths"]["/api/v2/laffat"]["get"]
    parameters = {item["name"] for item in operation["parameters"]}
    assert {
        "search", "sourceKind", "replayValid", "qualificationStatus", "truthLabel",
        "contract", "project", "tactic", "seat", "renderStatus", "sort", "offset", "limit",
    } <= parameters
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/UnifiedLaffaLibraryResponseV1")
    detail = app.openapi()["paths"]["/api/v2/laffat/{identifier}"]["get"]
    detail_schema = detail["responses"]["200"]["content"]["application/json"]["schema"]
    assert detail_schema["$ref"].endswith("/UnifiedLaffaRowV1")
    assert "/api/v2/laffat/{identifier}/evidence" in app.openapi()["paths"]


def test_research_federation_uses_the_shared_sqlite_postgres_query_contract():
    assert inspect.signature(SQLiteResearchStoreV1.list_results) == inspect.signature(
        PostgresResearchStoreV1.list_results
    )
    assert inspect.signature(
        SQLiteResearchStoreV1.list_result_library_projections
    ) == inspect.signature(PostgresResearchStoreV1.list_result_library_projections)
    parameters = inspect.signature(PostgresResearchStoreV1.list_results).parameters
    assert {"sort", "offset", "limit"} <= set(parameters)
    source = inspect.getsource(LaffaLibraryServiceV1)
    assert "list_result_library_projections" in source
    assert "SQLiteResearchStoreV1" not in source and "._db" not in source


def test_v4_library_uses_the_checksum_bound_compact_projection_without_detail_reads() -> None:
    compact = {
        "schemaVersion": 2,
        "resultCode": "LAF-V4-STALE",
        "resultHash": _digest("4"),
        "runCode": "RUN-V4-STALE",
        "disposition": "qualified",
        "publicationEligible": True,
        "qualification": {"qualified": True, "observerSafe": True},
    }
    compact["traceHash"] = _digest("5")
    compact["artifacts"] = {}
    compact["scenarioProjection"] = {"exactStateAvailable": True}

    class Store:
        calls = 0

        def list_result_library_projections(self, *, offset: int, limit: int):
            self.calls += 1
            assert offset == 0 and limit == 500
            return {"items": [compact], "total": 1, "nextOffset": None}

    research = SimpleNamespace(
        store=Store(),
        get_result=lambda _result_code: (_ for _ in ()).throw(
            AssertionError("the library must not deep-read every immutable result")
        ),
    )
    service = LaffaLibraryServiceV1(
        learning_store=None,
        research=research,
        foundry_catalogs=None,
        render_batches=None,
    )
    expected = [
        {
            **{
                key: compact.get(key)
                for key in (
                    "schemaVersion", "resultCode", "resultHash", "runCode",
                    "traceHash", "gameHash", "truthLabel", "confidence", "score",
                    "disposition", "publicationEligible", "qualificationFailures",
                    "qualification", "classification", "tacticalTags",
                    "verifiedTacticalTags", "interestingSeats", "libraryProjection",
                    "scenarioProjection", "createdAt",
                )
            },
            "artifacts": compact["artifacts"],
            "_libraryReplayValid": True,
        }
    ]
    assert service._research_results() == expected
    assert service._research_results() == expected
    assert research.store.calls == 2


def test_active_bundle_filter_skips_unreplayable_archives_before_deep_verification():
    class Learning(_Learning):
        deep_reads = 0

        def list_games(self, *, query: str = "", offset: int = 0, limit: int = 100):
            values = super().list_games(query=query, offset=offset, limit=limit)
            return [{**item, "bundleHash": _digest("b")} for item in values]

        def get_game(self, identifier: str):
            self.deep_reads += 1
            return super().get_game(identifier)

        def verify_game(self, identifier: str):
            self.deep_reads += 1
            return super().verify_game(identifier)

    learning = Learning()
    service = LaffaLibraryServiceV1(
        learning_store=learning,
        research=None,
        foundry_catalogs=None,
        render_batches=None,
        active_bundle_hash=_digest("a"),
    )
    response = service.list(limit=10)
    assert response["items"] == []
    assert learning.deep_reads == 0


@pytest.mark.parametrize("failure", [None, "identity", "ownership", "checksum"])
def test_native_library_detail_resolves_only_selected_public_roles(tmp_path, monkeypatch, failure):
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    payload = json.dumps({"exactNormalization": {"state": {
        "buyer": "west", "dealer": "south", "hands": {"east": ["private-unplayed-card"]},
        "finalScore": [99, 0],
    }}}).encode()
    reference = artifacts.put(payload, media_type="application/json")
    compact = {
        "schemaVersion": 2, "classification": "scenario_v4",
        "resultCode": "LAF-ROLE-DETAIL", "resultHash": _digest("4"),
        "runCode": "RUN-ROLE-DETAIL", "traceHash": _digest("5"),
        "disposition": "qualified", "publicationEligible": False,
        "scenarioProjection": {"exactStateAvailable": True},
        "artifacts": {"completeTrace": reference},
    }
    other = {**compact, "resultCode": "LAF-OTHER", "resultHash": _digest("6"),
             "traceHash": _digest("7")}
    calls = []

    def get_result(code):
        calls.append(code)
        assert code == compact["resultCode"]
        if failure == "ownership":
            raise RuntimeError("completeTrace substitution detected")
        return {**compact, "resultHash": _digest("0") if failure == "identity" else compact["resultHash"]}

    research = SimpleNamespace(
        store=SimpleNamespace(list_result_library_projections=lambda **_kwargs:
                              {"items": [compact, other], "total": 2, "nextOffset": None}),
        artifact_store=artifacts, get_result=get_result,
    )
    service = LaffaLibraryServiceV1(
        learning_store=SimpleNamespace(list_games=lambda **_kwargs: [], list_video_records=lambda **_kwargs: []),
        research=research, foundry_catalogs=None, render_batches=None,
    )
    get_artifact = artifacts.get
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "get", lambda _uri: pytest.fail("list queries must not read traces"))
        page = service.list(limit=2)
    assert calls == []
    original = next(item for item in page["items"] if item["humanCode"] == compact["resultCode"])
    assert original["buyer"] is original["dealer"] is None
    if failure == "checksum":
        monkeypatch.setattr(artifacts, "get", lambda uri: get_artifact(uri) + b" ")
    if failure in {"identity", "ownership"}:
        with pytest.raises(RuntimeError, match="identity mismatch|substitution detected"):
            service.get(compact["resultCode"])
    else:
        detail = service.get(compact["resultCode"])
        assert detail == {**original, **({"buyer": "west", "dealer": "south"} if failure is None else {})}
        assert "private-unplayed-card" not in json.dumps(detail)
        assert "finalScore" not in detail and "hands" not in detail
    assert calls == [compact["resultCode"]]
    assert service.list(limit=2) == page
