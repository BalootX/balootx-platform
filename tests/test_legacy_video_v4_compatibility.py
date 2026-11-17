from __future__ import annotations

from copy import deepcopy
import json

from fastapi.testclient import TestClient

from baloot_platform.service import api


class _FakeResearch:
    def __init__(self) -> None:
        self.created: list[tuple[dict, str]] = []
        self.submitted: list[tuple[str, str]] = []
        self.run = {
            "runCode": "RUN-ABCDEF123456",
            "recipeCode": "RCP-ABCDEF1234",
            "status": "queued",
            "progress": 0.0,
            "resultCount": 0,
            "qualifiedCount": 0,
            "quarantineCount": 0,
            "scenarioExecutorIdentityHash": "a" * 64,
        }

    def create_recipe(self, payload, *, idempotency_key):
        self.created.append((deepcopy(dict(payload)), idempotency_key))
        return {"code": "RCP-ABCDEF1234"}

    def submit(self, recipe_code, *, idempotency_key):
        self.submitted.append((recipe_code, idempotency_key))
        return deepcopy(self.run)

    def get_run(self, run_code):
        return deepcopy(self.run) if run_code == self.run["runCode"] else None

    def cancel(self, run_code):
        if run_code != self.run["runCode"]:
            raise KeyError(run_code)
        self.run["status"] = "cancelled"
        return deepcopy(self.run)

    def events(self, run_code):
        if run_code != self.run["runCode"]:
            raise KeyError(run_code)
        return [{"type": self.run["status"], "at": "2026-08-23T00:00:00Z"}]


def _predicate_map(recipe: dict) -> dict[str, dict]:
    rows: list[dict] = []

    def walk(value: dict) -> None:
        if "predicateId" in value:
            rows.append(value)
            return
        for child in value.get("children", []):
            walk(child)

    walk(recipe["scenario"]["constraints"])
    walk(recipe["scenario"]["outcome"])
    return {row["predicateId"]: row for row in rows}


def test_legacy_video_post_compiles_to_frozen_v4_research(monkeypatch):
    research = _FakeResearch()
    monkeypatch.setattr(api, "RESEARCH", research)
    monkeypatch.setattr(
        api,
        "_execute_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the direct scenario generator must not run")
        ),
    )
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/video-jobs",
        headers={"Idempotency-Key": "operator-request-17"},
        json={
            "bundleRevisionId": "CBR-FROZEN-TEST",
            "seed": 41,
            "dealer": "north",
            "contract": "sun",
            "buyer": "west",
            "matchScoreUs": 17,
            "matchScoreThem": 23,
            "requireProject": False,
            "pauseTrickMin": 5,
            "pauseTrickMax": 7,
            "maxSeedAttempts": 6,
            "maxCandidatesPerSeed": 1,
        },
    )

    assert response.status_code == 202
    assert response.headers["deprecation"] == "true"
    assert response.json() == {
        "jobId": "RUN-ABCDEF123456",
        "status": "queued",
        "progress": 0.0,
        "authority": "research_v4",
        "recipeCode": "RCP-ABCDEF1234",
        "bundleRevisionId": "CBR-FROZEN-TEST",
        "rendersAutomatically": False,
        "deprecation": {
            "schemaVersion": 1,
            "status": "deprecated",
            "successorEndpoints": api._LEGACY_VIDEO_V4_SUCCESSORS,
        },
    }
    assert len(research.created) == len(research.submitted) == 1
    recipe, recipe_key = research.created[0]
    predicates = _predicate_map(recipe)
    assert recipe["schemaVersion"] == 4
    assert recipe["configuration"] == {
        "schemaVersion": 1,
        "bundleRevisionId": "CBR-FROZEN-TEST",
    }
    assert recipe["scenario"]["origin"] == {"kind": "search"}
    assert recipe["scenario"]["decisionPoint"] == {
        "kind": "play",
        "selector": {"remainingCards": 7, "actingSeat": "south"},
    }
    assert predicates["PRED-LEGACY-CONTRACT"]["arguments"] == {
        "contract": "sun"
    }
    assert predicates["PRED-LEGACY-BUYER"]["arguments"] == {"seat": "west"}
    assert predicates["PRED-LEGACY-TRICK"]["arguments"] == {
        "minimum": 6,
        "maximum": 6,
    }
    assert predicates["PRED-LEGACY-LEGAL-ACTIONS"]["arguments"] == {
        "minimum": 2,
        "maximum": 8,
    }
    assert "PRED-LEGACY-PROJECT-US" not in predicates
    assert recipe["resources"]["seedStart"] == 41
    assert recipe["resources"]["seedCount"] == 6
    assert recipe["resources"]["seedsPerShard"] == 3
    assert not {
        "candidatesPerSeed",
        "maximumConflicts",
        "analysisMode",
    }.intersection(recipe["resources"])
    assert recipe["proof"]["mode"] == "exact"
    assert recipe["renderPolicy"] == {"mode": "on_demand"}
    assert recipe_key.startswith("legacy-video-recipe-")
    assert research.submitted[0][0] == "RCP-ABCDEF1234"
    assert research.submitted[0][1].startswith("legacy-video-run-")
    assert recipe_key.removeprefix("legacy-video-recipe-") == research.submitted[
        0
    ][1].removeprefix("legacy-video-run-")


def test_legacy_video_post_rejects_unmapped_or_unknown_fields_without_side_effects(
    monkeypatch,
):
    research = _FakeResearch()
    monkeypatch.setattr(api, "RESEARCH", research)
    client = TestClient(api.app)
    headers = {"Idempotency-Key": "operator-rejection-tests"}

    missing_key = client.post(
        "/api/v1/video-jobs",
        json={"bundleRevisionId": "CBR-FROZEN-TEST"},
    )
    assert missing_key.status_code == 428

    missing_revision = client.post(
        "/api/v1/video-jobs",
        headers=headers,
        json={"contract": "sun"},
    )
    assert missing_revision.status_code == 422
    assert missing_revision.json()["detail"]["code"] == "bundle_revision_required"

    presentation_override = client.post(
        "/api/v1/video-jobs",
        headers=headers,
        json={
            "bundleRevisionId": "CBR-FROZEN-TEST",
            "projectsEnabled": True,
            "theme": "emerald",
        },
    )
    assert presentation_override.status_code == 422
    assert presentation_override.json()["detail"]["code"] == (
        "legacy_video_overrides_unsupported"
    )
    assert presentation_override.json()["detail"]["unsupportedFields"] == [
        "projectsEnabled",
        "theme",
    ]

    escalation_override = client.post(
        "/api/v1/video-jobs",
        headers=headers,
        json={
            "bundleRevisionId": "CBR-FROZEN-TEST",
            "escalationLevel": "double",
        },
    )
    assert escalation_override.status_code == 422
    assert escalation_override.json()["detail"]["code"] == (
        "legacy_play_escalation_unsupported"
    )

    false_candidate_control = client.post(
        "/api/v1/video-jobs",
        headers=headers,
        json={
            "bundleRevisionId": "CBR-FROZEN-TEST",
            "maxCandidatesPerSeed": 2,
        },
    )
    assert false_candidate_control.status_code == 422
    assert false_candidate_control.json()["detail"]["code"] == (
        "legacy_video_candidates_per_seed_unsupported"
    )

    for pause_minimum, pause_maximum in ((7, 7), (0, 3)):
        trick_range = client.post(
            "/api/v1/video-jobs",
            headers=headers,
            json={
                "bundleRevisionId": "CBR-FROZEN-TEST",
                "pauseTrickMin": pause_minimum,
                "pauseTrickMax": pause_maximum,
            },
        )
        assert trick_range.status_code == 422
        assert trick_range.json()["detail"]["code"] == (
            "legacy_video_trick_range_unsupported"
        )

    unknown = client.post(
        "/api/v1/video-jobs",
        headers=headers,
        json={"bundleRevisionId": "CBR-FROZEN-TEST", "oracleRepair": True},
    )
    assert unknown.status_code == 422

    coercive = client.post(
        "/api/v1/video-jobs",
        headers=headers,
        json={"bundleRevisionId": "CBR-FROZEN-TEST", "seed": "41"},
    )
    assert coercive.status_code == 422
    assert research.created == []
    assert research.submitted == []


def test_legacy_video_research_job_reads_cancels_and_lists_events(monkeypatch):
    research = _FakeResearch()
    monkeypatch.setattr(api, "RESEARCH", research)
    original_events = research.events
    private_event = {"type": "progress", "sequence": 2, "currentStage": "family_proposal",
        "screenedSeed": 6, "screenedDealer": "south",
        "auctionQualification": {"status": "complete", "contract": "sun", "allPassed": False},
        "incompleteCompilation": {"futurePlays": ["SA"], "roundScore": [64, 6]}}
    monkeypatch.setattr(research, "events", lambda code: [*original_events(code), private_event])
    client = TestClient(api.app)
    run_code = research.run["runCode"]

    read = client.get(f"/api/v1/video-jobs/{run_code}")
    assert read.status_code == 200
    assert read.headers["deprecation"] == "true"
    assert read.json()["authority"] == "research_v4"
    assert read.json()["rendersAutomatically"] is False

    events = client.get(f"/api/v1/video-jobs/{run_code}/events")
    assert events.status_code == 200
    assert events.json()["events"][0]["type"] == "queued"
    assert events.json()["events"][1] == {"type": "progress", "sequence": 2, "currentStage": "family_proposal"}
    assert research.events(run_code)[1] == private_event

    cancelled = client.post(f"/api/v1/video-jobs/{run_code}:cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert research.run["status"] == "cancelled"


def test_legacy_video_routes_are_marked_deprecated_with_v4_successors():
    document = api.app.openapi()
    operation = document["paths"]["/api/v1/video-jobs"]["post"]
    assert operation["deprecated"] is True
    assert operation["x-baloot-authority"] == "research_v4"
    assert operation["x-baloot-successor-endpoints"] == list(
        api._LEGACY_VIDEO_V4_SUCCESSORS.values()
    )
    assert operation["parameters"][0]["name"] == "Idempotency-Key"
    assert operation["parameters"][0]["required"] is True


def test_legacy_video_options_expose_the_exact_compatibility_boundary():
    response = TestClient(api.app).get("/api/v1/options")
    assert response.status_code == 200
    compatibility = response.json()["legacyVideoCompatibility"]
    assert compatibility["deprecated"] is True
    assert compatibility["authority"] == "research_v4"
    assert compatibility["requiredFields"] == ["bundleRevisionId"]
    assert compatibility["requiredHeaders"] == ["Idempotency-Key"]
    assert compatibility["supportedScopes"] == ["round"]
    assert compatibility["supportedRulesProfiles"] == ["full"]
    assert compatibility["successorEndpoints"]["results"].endswith(
        "?runCode={runCode}"
    )


def test_legacy_video_proxy_does_not_relabel_older_research_runs(monkeypatch):
    research = _FakeResearch()
    research.run["scenarioExecutorIdentityHash"] = None
    monkeypatch.setattr(api, "RESEARCH", research)

    response = TestClient(api.app).get(
        f"/api/v1/video-jobs/{research.run['runCode']}"
    )

    assert response.status_code == 404


def test_default_legacy_adapter_recipe_is_canonically_valid_and_admissible():
    published_revision = api.CONTAINER.configuration.resolve()["bundle"][
        "revisionId"
    ]
    request = api.VideoJobRequest(
        bundleRevisionId=published_revision,
        maxSeedAttempts=1,
        maxCandidatesPerSeed=1,
    )
    recipe = api._legacy_video_v4_recipe(request)

    validated = api.RESEARCH.validate_recipe(recipe)
    estimate = api.RESEARCH.estimate(recipe)

    assert validated["valid"] is True
    assert validated["recipe"]["schemaVersion"] == 4
    assert validated["configurationAuthority"]["bundleRevisionId"] == (
        request.bundleRevisionId
    )
    assert estimate["launchAdmissible"] is True
    assert estimate["proof"]["remainingCardCount"] == 7
    assert estimate["proof"]["requestedMode"] == "exact"




def test_grpc_create_video_uses_the_shared_v4_adapter():
    source = (api.ROOT / "src/baloot_platform/service/grpc_service.py").read_text(
        encoding="utf-8"
    )
    create_video = source.split("async def CreateVideo", 1)[1].split(
        "async def GetJob", 1
    )[0]
    assert "_submit_legacy_video_compatibility" in create_video
    assert "_execute_job" not in create_video
    assert "STORE.create" not in create_video
    assert "idempotency-key" in create_video


def test_mcp_create_video_describes_and_forwards_the_v4_compatibility_boundary():
    source = (api.ROOT / "src/baloot_platform/service/mcp_server.py").read_text(
        encoding="utf-8"
    )
    tool = source.split('"name": "create_video"', 1)[1].split(
        '"name": "get_video_job"', 1
    )[0]
    handler = source.split('if name == "create_video":', 1)[1].split(
        'if name == "get_video_job":', 1
    )[0]
    assert "Research V4" in tool
    assert "does not render automatically" in tool
    assert '"idempotency_key"' in tool
    assert '"idempotency-key"' in handler
    assert "_require_trusted()" in handler


def test_legacy_video_capability_and_docs_declare_research_render_separation():
    import yaml

    root = api.ROOT
    source = yaml.safe_load(
        (root / "configs/capabilities/platform_v20.yaml").read_text(
            encoding="utf-8"
        )
    )["operations"]["premium_video"]
    packaged = yaml.safe_load(
        (
            root
            / "src/baloot_platform/resources/configs/capabilities/platform_v20.yaml"
        ).read_text(encoding="utf-8")
    )["operations"]["premium_video"]
    assert source == packaged
    assert source["lifecycle"] == "deprecated"
    assert source["authority"] == "research_v4"
    assert source["output"] == "research_run"
    assert source["renders_automatically"] is False
    assert source["task"] == "durable_research_v4_compatibility"
    assert source["successors"]["prepare_render"].endswith(":prepare-render")
    assert source["successors"]["render"] == "/api/v2/render-batches"

    generated = json.loads(
        (root / "generated/capability-parity-v20.json").read_text(
            encoding="utf-8"
        )
    )["operations"]["premium_video"]
    assert generated["passed"] is True
    assert generated["contract"] == source

