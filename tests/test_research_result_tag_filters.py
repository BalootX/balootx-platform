from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest

from baloot_platform.research import postgres_store
from baloot_platform.research.family_catalog import TAG_AXES, TagQueryV1
from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.result_library import (
    ResearchResultQueryV1,
    result_facets,
    result_library_projection,
    stored_result_library_projection_hash_v1,
    stored_result_payload_hash_v1,
    validate_stored_result_library_projection_hash_v1,
    validate_result_lesson_tags_v1,
)
from baloot_platform.research.scenario_contracts import canonical_sha256
from baloot_platform.research.store import SQLiteResearchStoreV1


def _tags(**selections):
    core = {
        "schemaVersion": 1,
        "tags": {axis: selections.get(axis, []) for axis in TAG_AXES},
        "sources": {"proofArtifactHash": "a" * 64, "exactStateId": "STATE-FOCAL"},
        "playIndex": 20,
        "qualificationEvidence": False,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _result(code, tags=None):
    return {
        "resultCode": code,
        "resultHash": canonical_sha256(code),
        "runCode": "RUN-TAGS",
        "truthLabel": "exact_optimal",
        "confidence": 1.0,
        "score": 100.0,
        "disposition": "qualified",
        "createdAt": "2026-10-13T00:00:00+00:00",
        "scenarioProjection": {"lessonTags": tags} if tags is not None else {},
        # Requested family capabilities are not evidence for lesson facets.
        "capabilityTags": {axis: list(values) for axis, values in TAG_AXES.items()},
        "tacticalTags": ["round-flip", "advanced"],
    }


def test_generation_result_binding_rejects_unowned_requests_and_unproved_tags(root):
    from baloot_platform.research.family_catalog import ResearchFamilyCatalogV1, build_family_generation_request_v1
    from baloot_platform.research.result_authority import validate_generation_result_binding_v1
    catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').to_dict()
    request = build_family_generation_request_v1(dict(schemaVersion=1,
        catalogHash=catalog['sha256'], tagQuery={'impact': ['round-flip']}))
    recipe = {'generation': request}
    snapshot = {'components': {'challenge': {'payload': {'scenarioResearchV4': {
        'qualityExecution': {'familyCatalog': catalog}}}}}}
    trace = {'generationRequest': request, 'lessonTags': _tags(impact=['round-flip'])}
    validate_generation_result_binding_v1(recipe, trace, snapshot, qualified=True)
    validate_generation_result_binding_v1({}, {}, None, qualified=False)
    incomplete = {'status': 'trace_not_available', 'generationRequest': request}
    validate_generation_result_binding_v1(recipe, incomplete, snapshot, qualified=False)
    for owning_recipe, document, frozen in (
        ({}, trace, snapshot), (recipe, trace, None),
        (recipe, {**trace, 'generationRequest': {**request, 'catalogHash': '0' * 64}}, snapshot),
        (recipe, {**trace, 'lessonTags': _tags(impact=['points-only'])}, snapshot),
        (recipe, incomplete, snapshot), (recipe, {}, snapshot),
    ):
        with pytest.raises(ValueError, match='generation'):
            validate_generation_result_binding_v1(owning_recipe, document, frozen, qualified=True)


def _seed(store):
    store.put_recipe({"code": "RCP-TAGS", "sha256": "b" * 64, "status": "published"})
    store.create_run({
        "runCode": "RUN-TAGS", "runHash": "c" * 64,
        "recipeCode": "RCP-TAGS", "recipeHash": "b" * 64,
        "status": "completed", "progress": 1.0,
    })
    for code, tags in (
        ("LAF-A", _tags(impact=["round-flip"], technique=["unblock"],
                        level=["advanced"], mode=["sun"], trickBand=["trick-6"])),
        ("LAF-B", _tags(impact=["points-only"], technique=["ace-timing"],
                        level=["beginner"], mode=["sun"], trickBand=["trick-6"])),
        ("LAF-C", _tags(impact=["round-flip"], technique=["unblock"],
                        level=["advanced"], mode=["hokum"], trickBand=["trick-6"])),
        ("LAF-D", _tags(mode=["sun"], trickBand=["trick-6"])),
        ("LAF-LEGACY", None),
    ):
        store.put_result(_result(code, tags))


def test_lesson_facets_use_only_validated_proven_values_and_preserve_empty_axes():
    document = _tags(impact=["points-only"], mode=["sun"], trickBand=["trick-6"])
    payload = _result("LAF-TAGS", document)
    facets = result_facets(payload)
    assert {key: value for key, value in facets.items() if key.startswith("lesson_tag_")} == {
        "lesson_tag_impact": ("points-only",),
        "lesson_tag_mode": ("sun",),
        "lesson_tag_trickBand": ("trick-6",),
    }
    assert result_library_projection(payload, created_at=payload["createdAt"])[
        "scenarioProjection"
    ]["lessonTags"] == document
    assert document["tags"]["technique"] == document["tags"]["level"] == []
    assert not any(key.startswith("lesson_tag_") for key in result_facets(_result("LAF-OLD")))


@pytest.mark.parametrize("classifier", [None, {"family": "overtake_unblock", "proofArtifactHash": "a" * 64}])
def test_trace_derived_lesson_document_retains_optional_provenance(classifier):
    document = _tags(technique=["unblock"], mode=["sun"], trickBand=["trick-6"])
    document.update(classifier=classifier, qualityEvidenceHash="d" * 64)
    document["sha256"] = canonical_sha256({key: item for key, item in document.items() if key != "sha256"})
    assert validate_result_lesson_tags_v1(document).to_dict() == {
        "technique": ["unblock"], "mode": ["sun"], "trickBand": ["trick-6"],
    }
    payload = _result("LAF-TRACE-TAGS", document)
    assert result_facets(payload)["lesson_tag_technique"] == ("unblock",)
    assert result_library_projection(payload, created_at=payload["createdAt"])["scenarioProjection"]["lessonTags"] == document


@pytest.mark.parametrize("field,value", [
    ("schemaVersion", True), ("schemaVersion", 2),
    ("qualificationEvidence", True), ("sources", []),
    ("playIndex", True), ("playIndex", 15), ("playIndex", 28),
    ("tags", []), ("extra", "unrecognized"),
    ("classifier", []), ("qualityEvidenceHash", None),
    ("qualityEvidenceHash", "not-sha256"), ("qualityEvidenceHash", "z" * 64),
])
def test_lesson_facets_reject_invalid_document_schema_even_when_rehashed(field, value):
    document = _tags(mode=["sun"])
    document[field] = value
    document["sha256"] = canonical_sha256({key: item for key, item in document.items() if key != "sha256"})
    with pytest.raises(ValueError, match="lesson tags document"):
        result_facets(_result("LAF-BAD", document))


@pytest.mark.parametrize("axis,values", [
    ("unknown", []), ("mode", "sun"), ("mode", [1]),
    ("mode", ["sun", "sun"]), ("mode", ["unknown"]),
    ("technique", ["round-flip"]),
])
def test_lesson_facets_reject_invalid_axis_values_even_when_rehashed(axis, values):
    document = _tags(**{axis: values})
    if axis not in TAG_AXES:
        document["tags"][axis] = values
        document["sha256"] = canonical_sha256({key: item for key, item in document.items() if key != "sha256"})
    with pytest.raises(ValueError, match="lesson tags axes|tag query"):
        result_facets(_result("LAF-BAD", document))


def test_lesson_facets_reject_document_hash_tampering():
    document = _tags(mode=["sun"])
    document["tags"]["mode"] = ["hokum"]
    with pytest.raises(ValueError, match="lesson tags document"):
        result_facets(_result("LAF-BAD", document))


@pytest.mark.parametrize("query", [{"impact": []}, {"unknown": ["sun"]},
    {"mode": ["sun", "sun"]}, {"mode": ["SUN"]}, {"mode": "sun"}])
def test_result_tag_query_uses_the_closed_shared_catalog(query):
    with pytest.raises(ValueError, match="tag query"):
        ResearchResultQueryV1(tag_query=query)


def test_result_tag_query_is_detached_and_canonical():
    raw = {"mode": ["hokum", "sun"], "impact": ["points-only", "round-flip"]}
    query = ResearchResultQueryV1(tag_query=raw)
    raw["mode"].clear()
    assert isinstance(query.tag_query, TagQueryV1)
    assert query.tag_facet_filters == (
        ("lesson_tag_impact", ("round-flip", "points-only")),
        ("lesson_tag_mode", ("sun", "hokum")),
    )
    assert ResearchResultQueryV1(tag_query={}).tag_facet_filters == ()
    assert ResearchResultQueryV1().tag_facet_filters == ()


def test_sqlite_lesson_tag_filters_and_axes_or_values_page_and_survive_reopen(tmp_path):
    path = tmp_path / "research.sqlite3"
    store = SQLiteResearchStoreV1(path)
    try:
        _seed(store)
        assert store.list_results(tag_query={})["total"] == 5
        queries = [
            ({"impact": ["round-flip", "points-only"], "mode": ["sun"]}, ["LAF-A", "LAF-B"]),
            ({"technique": ["unblock"], "level": ["advanced"], "trickBand": ["trick-6"]}, ["LAF-A", "LAF-C"]),
            ({"level": ["beginner"], "mode": ["hokum"]}, []),
            ({"mode": ["sun"]}, ["LAF-A", "LAF-B", "LAF-D"]),
        ]
        for query, expected in queries:
            response = store.list_results(tag_query=query, sort="result_code_asc")
            assert [row["resultCode"] for row in response["items"]] == expected
            assert response["total"] == len(expected)
        page = store.list_results(tag_query=queries[0][0], sort="result_code_asc", limit=1)
        assert page["total"] == 2 and page["nextOffset"] == 1 and page["hasMore"]
        second = store.list_results(tag_query=queries[0][0], sort="result_code_asc", offset=1, limit=1)
        assert [row["resultCode"] for row in second["items"]] == ["LAF-B"]
        assert second["nextOffset"] is None and not second["hasMore"]
        projected = {row["resultCode"]: row for row in store.list_result_library_projections()["items"]}
        assert projected["LAF-A"]["scenarioProjection"]["lessonTags"] == store.get_result("LAF-A")["scenarioProjection"]["lessonTags"]
        assert projected["LAF-D"]["scenarioProjection"]["lessonTags"]["tags"]["impact"] == []
        original = store.get_result("LAF-A")
    finally:
        store.close()
    reopened = SQLiteResearchStoreV1(path)
    try:
        assert reopened.list_results(tag_query={"impact": ["round-flip"]})["total"] == 2
        assert reopened.get_result("LAF-A") == original
        assert reopened.get_result("LAF-LEGACY")["scenarioProjection"] == {}
    finally:
        reopened.close()


def test_sqlite_invalid_lesson_document_rolls_back_result_and_facets(tmp_path):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    try:
        _seed(store)
        document = _tags(mode=["sun"])
        document["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="lesson tags document"):
            store.put_result(_result("LAF-BAD", document))
        assert store.get_result("LAF-BAD") is None
        assert store.list_results()["total"] == 5
        assert store._db.execute("SELECT COUNT(*) FROM research_result_facets WHERE result_code=?", ("LAF-BAD",)).fetchone()[0] == 0
    finally:
        store.close()


def test_lesson_tag_compact_projection_remains_bound_to_the_immutable_payload():
    payload = _result("LAF-TAGS", _tags(mode=["sun"]))
    projection = result_library_projection(payload, created_at=payload["createdAt"])
    payload_hash = stored_result_payload_hash_v1(payload)
    projection_hash = stored_result_library_projection_hash_v1(projection, payload_hash=payload_hash)
    tampered = copy.deepcopy(projection)
    tampered["scenarioProjection"]["lessonTags"]["tags"]["mode"] = ["hokum"]
    with pytest.raises(RuntimeError, match="library projection mismatch"):
        validate_stored_result_library_projection_hash_v1(
            tampered, result_code=payload["resultCode"], result_hash=payload["resultHash"],
            run_code=payload["runCode"], payload_hash=payload_hash, projection_hash=projection_hash,
        )


def test_postgres_tag_filter_sql_uses_parameterized_or_membership_per_and_axis(monkeypatch):
    calls = []

    class Rows:
        def fetchone(self):
            return {"count": 0}

        def fetchall(self):
            return []

    class Connection:
        def execute(self, statement, parameters):
            calls.append((statement, list(parameters)))
            return Rows()

    monkeypatch.setattr(postgres_store, "contextual_connection", lambda _pool: nullcontext(Connection()))
    store = object.__new__(PostgresResearchStoreV1)
    store.pool = object()
    result = store.list_results(
        tag_query={"mode": ["sun"], "impact": ["points-only", "round-flip"]},
        tactical_tag="unblock", offset=1, limit=3,
    )
    assert result["items"] == [] and result["total"] == 0
    for statement, parameters in calls:
        assert "tag0.facet_value IN (%s,%s)" in statement
        assert "tag1.facet_value IN (%s)" in statement
        assert "AND EXISTS" in statement
        assert "round-flip" not in statement and "points-only" not in statement
        assert parameters[:7] == ["tactical_tag", "unblock", "lesson_tag_impact",
                                  "round-flip", "points-only", "lesson_tag_mode", "sun"]
    assert calls[1][1][-2:] == [3, 1]


def test_postgres_tag_facet_batch_uses_the_same_proven_values():
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def executemany(self, _statement, rows):
            self.rows = list(rows)

    class Connection:
        def __init__(self):
            self.batch = Cursor()

        def execute(self, _statement, _parameters):
            return None

        def cursor(self):
            return self.batch

    connection = Connection()
    payload = _result("LAF-TAGS", _tags(mode=["sun"], technique=["unblock"]))
    PostgresResearchStoreV1._replace_result_facets(connection, payload["resultCode"], payload)
    assert {row for row in connection.batch.rows if row[1].startswith("lesson_tag_")} == {
        ("LAF-TAGS", "lesson_tag_mode", "sun"),
        ("LAF-TAGS", "lesson_tag_technique", "unblock"),
    }
