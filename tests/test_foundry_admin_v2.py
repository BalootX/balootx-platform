from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI, Request
from service_test_client import OperatorClient as TestClient

from baloot_platform.service.routes.foundry_admin_v2 import (
    build_foundry_admin_v2_router,
)


class Campaigns:
    def get_campaign(self, campaign_hash):
        return {"campaignHash": campaign_hash} if campaign_hash == "c" * 64 else None

    def list_shards(self, campaign_hash):
        return [
            {"specHash": "a" * 64, "campaignHash": campaign_hash},
            {"specHash": "b" * 64, "campaignHash": campaign_hash},
        ]


class LearningStore:
    def __init__(self):
        self.transitions = []

    def list_games(self, *, query="", offset=0, limit=100):
        return [
            {"gameHash": "a" * 64, "serial": "LAF-A", "label": query, "createdAt": "2026-01-02T00:00:00Z"},
            {"gameHash": "b" * 64, "serial": "LAF-B", "label": query, "createdAt": "2026-01-01T00:00:00Z"},
        ][offset : offset + limit]

    def get_game(self, identifier):
        return {"gameHash": identifier, "state": {}, "verification": {"valid": True}}

    def verify_game(self, _identifier):
        return {"valid": True}

    def list_video_records(self, **_values):
        return []

    def list_datasets(self):
        return [{"id": "DS-1"}]

    def list_models(self):
        return [{"id": "MODEL-1"}]

    def list_challenges(self, *, status=None, limit=100):
        return [{"id": "CHL-1", "status": status}][:limit]

    def update_challenge_status(self, challenge_id, status):
        self.transitions.append((challenge_id, status))
        return {"id": challenge_id, "status": status}

    def govern_challenge_status(self, challenge_id, **values):
        self.transitions.append((challenge_id, values["target"]))
        return {"id": challenge_id, "status": values["target"]}


class Learning:
    def build_dataset(self, **values):
        return {"id": "DS-2", **values}

    def train_model(self, dataset_id, *, overrides, backend):
        return {
            "id": "MODEL-2",
            "datasetId": dataset_id,
            "overrides": overrides,
            "backend": backend,
        }


class Hitl:
    def list_cases(self, limit=100):
        return [{"caseId": "CASE-1"}][:limit]

    def create_case(self, decision_id, **values):
        return {"caseId": "CASE-2", "decisionId": decision_id, **values}


class Revision:
    def __init__(self, revision_id):
        self.revision_id = revision_id

    def to_dict(self, **_values):
        return {"revisionId": self.revision_id}


class ConfigurationStore:
    def list_profiles(self, **_values):
        return [Revision("PROFILE-1")]

    def list_bundles(self, **_values):
        return [Revision("BUNDLE-1")]


class Registry:
    policy = SimpleNamespace(to_dict=lambda: {"requireSignedExternal": True})

    def list(self, kind=None):
        return [{"id": "plugin", "kind": kind or "all"}]


class FoundryCatalogs:
    def __init__(self):
        self.values = {}

    def list_sakkas(self, **_values):
        return {"resourceKind": "foundry_sakka", "items": [], "nextCursor": None}

    def list(self, **_values):
        return {"items": [], "nextCursor": None}

    def create_natural_sakka(self, references, *, target):
        return {"record": {"catalogHash": "a" * 64}, "document": {"target": target}}

    def create_curated_sakka(self, selections, *, target):
        return {"record": {"catalogHash": "b" * 64}, "document": {"target": target}}

    def load_sakka(self, sakka_hash):
        if sakka_hash not in {"a" * 64, "b" * 64}:
            raise KeyError(sakka_hash)
        return object()

    def get(self, sakka_hash):
        return {"record": {"catalogHash": sakka_hash}, "document": {}}


class Identities:
    def resolve(self, _token):
        return None, None


class ContextSearches:
    def __init__(self):
        self.jobs = {}

    def submit(self, **values):
        job = {
            "jobId": "CTX-1",
            "kind": "critical_context_search_v1",
            "status": "queued",
            "request": values,
        }
        self.jobs[job["jobId"]] = job
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def cancel(self, job_id):
        if job_id not in self.jobs:
            return False
        self.jobs[job_id]["status"] = "cancelled"
        return True

    def events(self, job_id):
        return [{"type": self.jobs[job_id]["status"], "payload": {}}]


class SakkaGenerations(ContextSearches):
    def submit(self, **values):
        job = {
            "jobId": "SAKKA-GEN-1",
            "kind": "natural_sakka_generation_v1",
            "status": "queued",
            "request": values,
        }
        self.jobs[job["jobId"]] = job
        return job


def client() -> tuple[TestClient, LearningStore]:
    learning_store = LearningStore()
    app = FastAPI()

    @app.middleware("http")
    async def api_identity(request: Request, call_next):
        request.state.api_key_valid = True
        return await call_next(request)

    app.include_router(build_foundry_admin_v2_router(
        campaign_control=Campaigns(),
        learning_store=learning_store,
        learning=Learning(),
        hitl=Hitl(),
        configuration=SimpleNamespace(store=ConfigurationStore()),
        registry=Registry(),
        foundry_catalogs=FoundryCatalogs(),
        context_searches=ContextSearches(),
        sakka_generations=SakkaGenerations(),
        identities=Identities(),
        require_trusted=lambda _request: None,
        publication_policy={
            "reviewer_roles": ["service"],
            "publisher_roles": ["service"],
        },
    ))
    return TestClient(app), learning_store


def test_v2_admin_resources_are_first_class_and_cursor_bounded():
    api, _store = client()
    headers = {"x-api-key": "key", "x-baloot-subject-id": "ADMIN-1"}

    shards = api.get(
        "/api/v2/shards",
        params={"campaignHash": "c" * 64, "limit": 1},
        headers=headers,
    ).json()
    assert len(shards["items"]) == 1 and shards["nextCursor"] == "a" * 64
    laffat = api.get("/api/v2/laffat", params={"limit": 1}, headers=headers).json()
    assert laffat["resourceKind"] == "unified_laffa_library_v1"
    assert laffat["items"][0]["humanCode"] == "LAF-A"
    assert laffat["nextOffset"] == 1
    filtered_laffat = api.get(
        "/api/v2/laffat",
        params={
            "sourceKind": "archived_game",
            "replayValid": "true",
            "sort": "code_asc",
            "offset": 1,
            "limit": 1,
        },
        headers=headers,
    ).json()
    assert filtered_laffat["items"][0]["humanCode"] == "LAF-B"
    assert filtered_laffat["sourceFacets"] == [
        {"sourceKind": "archived_game", "count": 2},
        {"sourceKind": "qualified_laffa", "count": 0},
        {"sourceKind": "research_audit_trace", "count": 0},
        {"sourceKind": "research_result", "count": 0},
        {"sourceKind": "sakka_round", "count": 0},
    ]
    assert api.get("/api/v2/sakkas", headers=headers).json()["resourceKind"] == "foundry_sakka"
    assert api.get("/api/v2/config-revisions", headers=headers).json()["items"][0]["revisionId"] == "PROFILE-1"
    assert api.get("/api/v2/plugins", headers=headers).json()["items"][0]["id"] == "plugin"

    context = api.post(
        "/api/v2/context-searches",
        headers={**headers, "idempotency-key": "ctx-1"},
        json={
            "parentTraceReference": {
                "ownerKind": "complete_trace_v1",
                "ownerHash": "a" * 64,
                "uri": "cas://sha256/" + "b" * 64,
                "sha256": "b" * 64,
                "bytes": 100,
                "mediaType": "application/vnd.balootx.complete-trace-v1+json",
            },
            "decisionId": "DEC-1",
            "contextProofHashes": ["c" * 64],
        },
    )
    assert context.status_code == 202 and context.json()["jobId"] == "CTX-1"
    assert api.get("/api/v2/context-searches/CTX-1", headers=headers).status_code == 200

    generated = api.post(
        "/api/v2/sakka-generations",
        headers={**headers, "idempotency-key": "natural-sakka-1"},
        json={"rootSeed": 20260803, "initialDealer": "east"},
    )
    assert generated.status_code == 202
    assert generated.json()["jobId"] == "SAKKA-GEN-1"
    assert api.get(
        "/api/v2/sakka-generations/SAKKA-GEN-1", headers=headers
    ).status_code == 200

    natural = api.post(
        "/api/v2/sakkas/natural",
        headers=headers,
        json={
            "target": 152,
            "traceReferences": [{
                "ownerKind": "complete_trace_v1",
                "ownerHash": "a" * 64,
                "uri": "cas://sha256/" + "b" * 64,
                "sha256": "b" * 64,
                "bytes": 100,
                "mediaType": "application/vnd.balootx.complete-trace-v1+json",
            }],
        },
    )
    assert natural.status_code == 201
    curated = api.post(
        "/api/v2/sakkas/curated",
        headers=headers,
        json={
            "target": 152,
            "selections": [{
                "contextSearchResultHash": "c" * 64,
                "contextualizedLaffaHash": "d" * 64,
            }],
        },
    )
    assert curated.status_code == 201
    assert api.get("/api/v2/sakkas/" + "a" * 64, headers=headers).status_code == 200


def test_v2_learning_corrections_and_publications_delegate_to_authorities():
    api, store = client()
    headers = {"x-api-key": "key", "x-baloot-subject-id": "ADMIN-1"}

    correction = api.post(
        "/api/v2/corrections",
        headers=headers,
        json={"decisionId": "DEC-1", "correctionType": "analysis_defect"},
    )
    assert correction.status_code == 201
    assert correction.json()["created_by"] == "ADMIN-1"
    dataset = api.post("/api/v2/datasets", headers=headers, json={})
    assert dataset.status_code == 201 and dataset.json()["id"] == "DS-2"
    model = api.post(
        "/api/v2/models",
        headers=headers,
        json={"datasetId": "DS-1", "backend": "numpy", "epochs": 2},
    )
    assert model.status_code == 202 and model.json()["overrides"]["epochs"] == 2
    publication = api.post(
        "/api/v2/publications",
        headers=headers,
        json={"challengeId": "CHL-1", "action": "review"},
    )
    assert publication.status_code == 200
    assert store.transitions == [("CHL-1", "review")]


def test_application_openapi_contains_required_v2_resource_roots():
    from baloot_platform.service.api import app

    paths = set(app.openapi()["paths"])
    assert {
        "/api/v2/shards",
        "/api/v2/laffat",
        "/api/v2/sakkas",
        "/api/v2/config-revisions",
        "/api/v2/plugins",
        "/api/v2/corrections",
        "/api/v2/datasets",
        "/api/v2/models",
        "/api/v2/publications",
        "/api/v2/context-searches",
        "/api/v2/context-searches/{job_id}",
        "/api/v2/sakkas/natural",
        "/api/v2/sakkas/curated",
        "/api/v2/sakkas/{sakka_hash}",
        "/api/v2/sakka-generations",
        "/api/v2/sakka-generations/{job_id}",
    } <= paths
