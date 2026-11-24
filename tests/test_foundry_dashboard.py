from __future__ import annotations

import json
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest
import yaml

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.research.contracts import research_recipe_configuration_v1


DashboardSourceModule = tuple[str, str]


def _authored_dashboard_source_modules(
    source_root: Path,
) -> tuple[DashboardSourceModule, ...]:
    modules: list[DashboardSourceModule] = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx"}:
            continue
        relative_path = path.relative_to(source_root)
        if relative_path.parts[:2] == ("api", "generated"):
            continue
        modules.append((relative_path.as_posix(), path.read_text(encoding="utf-8")))
    return tuple(sorted(modules, key=lambda item: item[0]))


def _dashboard_source_corpus(
    modules: tuple[DashboardSourceModule, ...],
) -> str:
    return "\n".join(
        f"/* authored-source:{module_path} */\n{source}"
        for module_path, source in modules
    )


def _assert_dashboard_source_order(
    modules: tuple[DashboardSourceModule, ...],
    *anchors: str,
) -> str:
    if len(anchors) < 2 or any(not anchor for anchor in anchors):
        raise AssertionError(
            "source-order checks require at least two non-empty anchors"
        )

    owners = [
        (module_path, source)
        for module_path, source in modules
        if all(anchor in source for anchor in anchors)
    ]
    if not owners:
        raise AssertionError(
            f"source-order anchors do not share an owning module: {anchors!r}"
        )
    if len(owners) != 1:
        raise AssertionError(
            "source-order anchors have ambiguous owners: "
            + ", ".join(module_path for module_path, _ in owners)
        )

    owner_path, owner_source = owners[0]
    cursor = 0
    for anchor in anchors:
        positions = [
            match.start()
            for match in re.finditer(re.escape(anchor), owner_source[cursor:])
        ]
        if not positions:
            raise AssertionError(
                f"source-order anchor is missing or out of order in {owner_path}: "
                f"{anchor!r}"
            )
        if len(positions) != 1:
            raise AssertionError(
                f"source-order anchor is ambiguous in {owner_path}: {anchor!r}"
            )
        cursor += positions[0] + len(anchor)
    return owner_path


def test_authored_dashboard_source_modules_are_deterministic_and_exclude_only_generated_api(
    tmp_path,
):
    source_root = tmp_path / "src"
    authored = {
        "z.ts": "export const z = 'z';",
        "a.tsx": "export const A = () => null;",
        "api/manual.ts": "export const manual = true;",
        "feature/generated/kept.ts": "export const kept = true;",
        "api/generated/client.ts": "export const excluded = true;",
        "ignored.js": "export const ignored = true;",
    }
    for relative_path, source in authored.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    first = _authored_dashboard_source_modules(source_root)
    second = _authored_dashboard_source_modules(source_root)

    assert first == second
    assert [module_path for module_path, _ in first] == [
        "a.tsx",
        "api/manual.ts",
        "feature/generated/kept.ts",
        "z.ts",
    ]
    corpus = _dashboard_source_corpus(first)
    assert "manual = true" in corpus
    assert "kept = true" in corpus
    assert "excluded = true" not in corpus
    assert "ignored = true" not in corpus


def test_dashboard_source_order_fails_closed_when_owner_is_missing(tmp_path):
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "only.ts").write_text("const first = true;", encoding="utf-8")
    modules = _authored_dashboard_source_modules(source_root)

    with pytest.raises(AssertionError, match="do not share an owning module"):
        _assert_dashboard_source_order(modules, "first", "missing")


def test_dashboard_source_order_fails_closed_when_owner_is_ambiguous(tmp_path):
    source_root = tmp_path / "src"
    source_root.mkdir()
    for name in ("first.ts", "second.tsx"):
        (source_root / name).write_text("start(); finish();", encoding="utf-8")
    modules = _authored_dashboard_source_modules(source_root)

    with pytest.raises(AssertionError, match="ambiguous owners"):
        _assert_dashboard_source_order(modules, "start();", "finish();")


def test_dashboard_source_order_fails_closed_when_anchor_is_ambiguous(tmp_path):
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "only.ts").write_text(
        "start(); duplicate(); duplicate(); finish();", encoding="utf-8"
    )
    modules = _authored_dashboard_source_modules(source_root)

    with pytest.raises(AssertionError, match="anchor is ambiguous"):
        _assert_dashboard_source_order(modules, "start();", "duplicate();", "finish();")


def _campaign(seed_start: int) -> SimulationCampaignSpecV1:
    return SimulationCampaignSpecV1.build(
        rules_hash="1" * 64,
        bundle_hash="2" * 64,
        engine_version="dashboard-test",
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release="equal-seat-test",
        seat_agents=("equal-seat-test",) * 4,
        lanes=("organic",),
        lane_profile_hashes={"organic": "6" * 64},
        seed_start=seed_start,
        seed_count=4,
        seeds_per_shard=2,
    )


def test_campaign_listing_is_stable_and_cursor_driven(tmp_path):
    store = SQLiteCampaignControlStore(
        tmp_path / "dashboard-campaigns.sqlite3", maximum_attempts=3
    )
    identities = sorted(
        store.create_campaign(_campaign(seed_start))["campaignHash"]
        for seed_start in (100, 200, 300)
    )
    first = store.list_campaigns(limit=2)
    second = store.list_campaigns(after=first[-1]["campaignHash"], limit=2)
    assert [item["campaignHash"] for item in first] == identities[:2]
    assert [item["campaignHash"] for item in second] == identities[2:]


def test_foundry_runtime_and_campaign_list_are_available_to_trusted_operator(
    root, tmp_path, monkeypatch
):
    from baloot_platform.service import api

    store = SQLiteCampaignControlStore(
        tmp_path / "dashboard-api.sqlite3", maximum_attempts=3
    )
    campaign = store.create_campaign(_campaign(400))
    monkeypatch.setattr(api, "FOUNDRY_CAMPAIGNS", store)
    with TestClient(api.app) as client:
        runtime = client.get("/api/v2/foundry-profile")
        assert runtime.status_code == 200, runtime.text
        assert runtime.json()["rulesHash"]
        assert runtime.json()["staticLaneProfileHashes"]["organic"]
        assert (
            runtime.json()["guidedSearchProfile"]["agentProfileId"]
            == "foundry_balanced"
        )
        assert runtime.json()["guidedSearchProfile"]["candidatesPerSeed"] == 1
        assert runtime.json()["laneAgentReleases"]["guided"].startswith(
            "foundry_balanced@"
        )
        assert (
            runtime.json()["laneAgentReleases"]["guided"]
            == runtime.json()["agentRelease"]
        )
        assert (
            runtime.json()["laneAgentReleases"]["organic"]
            == runtime.json()["agentRelease"]
        )
        assert (
            runtime.json()["candidateResimulationAgentRelease"]
            == runtime.json()["agentRelease"]
        )
        research_profile = AgentProfileRegistry.from_bundle(root).get(
            "foundry_balanced"
        )
        assert research_profile.mistake_probability == 0.0
        assert research_profile.deception_probability == 0.0
        assert research_profile.play_search_mode == "paired_information_set"
        assert research_profile.play_search_budget > 0
        assert research_profile.bid_samples > 0
        assert set(runtime.json()["laneSeatAgents"]["guided"].values()) == {
            runtime.json()["laneAgentReleases"]["guided"]
        }
        assert (
            runtime.json()["organicScreeningProfile"]["sha256"]
            == runtime.json()["staticLaneProfileHashes"]["organic"]
        )
        assert runtime.json()["sakkaGenerationProfile"]["target"] == 152
        assert runtime.json()["videoProductionBatchProfile"]["defaultTarget"] == 10
        assert runtime.json()["videoProductionBatchProfile"]["maximumTarget"] == 100
        assert runtime.json()["videoProductionBatchProfile"]["laneOrder"] == ["guided"]
        assert runtime.json()["videoProductionBatchProfile"]["finalistConcurrency"] == 1
        assert (
            runtime.json()["videoProductionBatchProfile"]["renderPolicy"] == "on_demand"
        )
        assert runtime.json()["videoProductionBatchProfile"]["sha256"]
        executor = runtime.json()["decisionAnalysisExecutorProfile"]
        assert executor["screening"]["workerCount"] == 2
        assert executor["confirmation"]["workerCount"] == 2
        assert executor["finalistReanalysis"]["workerCount"] == 2
        configured_finalist = runtime.json()["finalistValidationProfile"][
            "agentProfileId"
        ]
        assert configured_finalist == "finalist_candidate"
        assert runtime.json()["finalistValidationProfile"]["maximumTracesPerJob"] == 4
        assert len(set(runtime.json()["finalistSeatAgents"].values())) == 1
        qualification = runtime.json()["finalistAgentQualification"]
        assert qualification["profileId"] == configured_finalist
        assert qualification["configuredEqual"] is True
        assert qualification["zeroMistakeSearchReady"] is True
        assert qualification["benchmarkEvidenceComplete"] is False
        assert qualification["professionalReviewComplete"] is False
        assert qualification["expertGenerationAllowed"] is False
        assert qualification["status"] == "configured_candidate"
        assert qualification["sha256"] == runtime.json()["finalistAgentRelease"]
        pilot = runtime.json()["finalistAgentPilotEvidence"]
        assert pilot["candidatePolicy"] == configured_finalist
        assert pilot["qualificationEligible"] is False
        assert pilot["status"] in {
            "passed",
            "missing",
            "stale",
            "invalid",
            "threshold_failed",
        }
        assert qualification["expertGenerationAllowed"] is False
        assert set(runtime.json()["finalistSeatAgents"].values()) == {
            qualification["sha256"]
        }
        assert runtime.json()["runtimeBundleRevisionId"]
        ready = client.get("/readyz")
        current = client.get("/api/v1/bundles/current")
        assert ready.json()["bundleHash"] == runtime.json()["bundleHash"]
        assert current.json()["sha256"] == runtime.json()["bundleHash"]

        listing = client.get("/api/v2/campaigns", params={"limit": 100})
        assert listing.status_code == 200, listing.text
        assert listing.json()["items"][0]["campaignHash"] == campaign["campaignHash"]
        assert listing.json()["nextCursor"] is None
        video_batches = client.get(
            "/api/v2/video-production-batches", params={"limit": 20}
        )
        assert video_batches.status_code == 200, video_batches.text
        assert isinstance(video_batches.json()["items"], list)


def test_legacy_operator_pages_redirect_to_the_single_foundry_journey():
    from baloot_platform.service import api

    with TestClient(api.app) as client:
        for route in ("/studio", "/challenges", "/decision-workbench", "/hitl-studio"):
            response = client.get(route, follow_redirects=False)
            assert response.status_code == 307
            assert response.headers["location"] == "/foundry"


def test_foundry_submits_target_twenty_using_only_governed_batch_profile(monkeypatch):
    from baloot_platform.service import api

    captured = {}

    class VideoBatchCapture:
        @staticmethod
        def submit(request, *, idempotency_key):
            captured["request"] = request
            captured["idempotencyKey"] = idempotency_key
            return {
                "jobId": "FVB-TWENTY",
                "tenantId": "default",
                "kind": "foundry_video_batch_v1",
                "status": "queued",
                "priority": 50,
                "progress": 0.0,
                "request": request,
                "result": None,
                "error": None,
                "attempts": 0,
                "maxAttempts": 3,
                "leaseOwner": None,
                "leaseUntil": None,
                "cancelRequested": False,
                "createdAt": "2026-08-05T00:00:00Z",
                "updatedAt": "2026-08-05T00:00:00Z",
            }

    monkeypatch.setattr(api, "FOUNDRY_VIDEO_PRODUCTION", VideoBatchCapture())
    profile = api.FOUNDRY_PROFILE["videoProductionBatchProfile"]
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v2/video-production-batches",
            headers={"Idempotency-Key": "ui-target-ten"},
            json={
                "targetCount": 10,
                "seedStart": 202608050001,
                "profileHash": profile["sha256"],
            },
        )

    assert response.status_code == 202, response.text
    assert captured["idempotencyKey"] == "ui-target-ten"
    assert captured["request"]["targetVideos"] == 10
    assert captured["request"]["roundsPerWave"] == profile["waveSeedCount"]
    assert captured["request"]["laneOrder"] == profile["laneOrder"]
    assert captured["request"]["finalistConcurrency"] == profile["finalistConcurrency"]
    assert captured["request"]["minimumConfidence"] == profile["minimumConfidence"]


def test_legacy_challenge_studio_converges_on_the_single_foundry_workflow():
    from baloot_platform.service import api

    with TestClient(api.app) as client:
        public = client.get("/", follow_redirects=False)
        assert public.status_code == 307
        assert public.headers["location"] == "/play"

        studio = client.get("/studio", follow_redirects=False)
        assert studio.status_code == 307
        assert studio.headers["location"] == "/foundry"

        response = client.get("/challenges", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/foundry"


def test_foundry_analysis_history_is_available_to_the_connected_dashboard(monkeypatch):
    from baloot_platform.service import api

    class AnalysisHistory:
        @staticmethod
        def list(*, status=None, limit=100):
            return [
                {
                    "jobId": "FDA-FRIENDLY",
                    "tenantId": "default",
                    "kind": "foundry_deep_analysis_v1",
                    "status": status or "completed",
                    "priority": 40,
                    "progress": 1.0,
                    "request": {},
                    "result": {"limit": limit},
                    "error": None,
                    "attempts": 1,
                    "maxAttempts": 3,
                    "leaseOwner": None,
                    "leaseUntil": None,
                    "cancelRequested": False,
                    "createdAt": "2026-08-03 12:00:00",
                    "updatedAt": "2026-08-03 12:01:00",
                }
            ]

    monkeypatch.setattr(api, "FOUNDRY_DEEP_ANALYSIS", AnalysisHistory())
    with TestClient(api.app) as client:
        response = client.get("/api/v2/analysis-batches", params={"limit": 20})
        assert response.status_code == 200
        assert response.json()["items"][0]["jobId"] == "FDA-FRIENDLY"


def test_local_foundry_execution_endpoints_connect_campaign_and_analysis_workers(
    tmp_path, monkeypatch
):
    from baloot_platform.service import api

    store = SQLiteCampaignControlStore(
        tmp_path / "dashboard-execution.sqlite3", maximum_attempts=3
    )
    campaign = store.create_campaign(_campaign(500))
    scheduled_campaigns: list[str] = []
    scheduled_analysis: list[bool] = []

    class AnalysisQueue:
        item = {
            "jobId": "FDA-CONNECTED",
            "tenantId": "default",
            "kind": "foundry_deep_analysis_v1",
            "status": "queued",
            "priority": 40,
            "progress": 0.0,
            "request": {"campaignHash": campaign["campaignHash"]},
            "result": None,
            "error": None,
            "attempts": 0,
            "maxAttempts": 3,
            "leaseOwner": None,
            "leaseUntil": None,
            "cancelRequested": False,
            "createdAt": "2026-08-05 12:00:00",
            "updatedAt": "2026-08-05 12:00:00",
        }

        def get(self, job_id):
            return self.item if job_id == self.item["jobId"] else None

    monkeypatch.setenv("BALOOT_ENV", "test")
    monkeypatch.setattr(api, "FOUNDRY_CAMPAIGNS", store)
    monkeypatch.setattr(api, "FOUNDRY_DEEP_ANALYSIS", AnalysisQueue())
    monkeypatch.setattr(api, "_recover_local_foundry_work", lambda: None)
    monkeypatch.setattr(
        api, "_schedule_local_foundry_campaign", scheduled_campaigns.append
    )
    monkeypatch.setattr(
        api, "_schedule_local_analysis_worker", lambda: scheduled_analysis.append(True)
    )

    with TestClient(api.app) as client:
        campaign_response = client.post(
            f"/api/v2/campaigns/{campaign['campaignHash']}:execute"
        )
        analysis_response = client.post(
            "/api/v2/analysis-batches/FDA-CONNECTED:execute"
        )

    assert campaign_response.status_code == 202, campaign_response.text
    assert analysis_response.status_code == 202, analysis_response.text
    assert scheduled_campaigns == [campaign["campaignHash"]]
    assert scheduled_analysis == [True]


def test_generation_catalog_kind_is_valid_for_the_foundry_dashboard() -> None:
    from baloot_platform.service.api import FoundryCatalogRecordResponse

    validated = FoundryCatalogRecordResponse.model_validate(
        {
            "catalogHash": "a" * 64,
            "kind": "natural_sakka_generation_v1",
            "artifact": {"sha256": "b" * 64},
            "metadata": {"roundCount": 8},
            "createdAt": "2026-08-03 12:00:00",
        }
    )
    assert validated.kind == "natural_sakka_generation_v1"


def test_ortools_scenario_studio_binds_final_catalog_and_profile_copy(root) -> None:
    from baloot_platform.research.scenario_predicate_manifest_v2 import (
        ScenarioPredicateCatalogLoweringManifestV2,
    )

    manifest = ScenarioPredicateCatalogLoweringManifestV2()
    definitions = manifest.exposed_definitions
    assert manifest.manifest_hash == (
        "d36815fde96af7246bfb67153f75dd0cc7962fc5b18827a3e6582b764a9a600d"
    )
    assert len(definitions) == 78
    assert len({definition.kind for definition in definitions}) == 78

    product = yaml.safe_load(
        (root / "configs" / "product" / "studio_ortools.yaml").read_text(
            encoding="utf-8"
        )
    )
    workflow = product["foundry_copy"]["groups"]["workflow"]
    for key in (
        "research_scenario_catalog",
        "research_scenario_catalog_help",
        "research_scenario_visibility",
        "research_scenario_visibility_complete_state",
        "research_scenario_visibility_actor_information_set",
        "research_scenario_authority_title",
        "research_scenario_authority_help",
    ):
        assert workflow[key]

    def schema_copy_keys(value):
        if isinstance(value, dict):
            for key in ("x-messageArKey", "x-messageEnKey"):
                if key in value:
                    yield value[key]
            for key in ("x-enumMessageArKeys", "x-enumMessageEnKeys"):
                if isinstance(value.get(key), dict):
                    yield from value[key].values()
            for child in value.values():
                yield from schema_copy_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from schema_copy_keys(child)

    for definition in definitions:
        assert workflow[definition.message_ar_key]
        assert workflow[definition.message_en_key]
        assert all(workflow[key] for key in schema_copy_keys(definition.argument_schema))

    observer = next(
        definition
        for definition in definitions
        if definition.kind == "state.observer_visibility"
    )
    assert observer.allowed_hardness == ("hard", "report")
    assert observer.cp_sat_support.actor_information_set == "nonanticipative"

    source = (
        root
        / "src"
        / "baloot_platform"
        / "service"
        / "foundry_dashboard"
        / "src"
        / "features"
        / "research"
        / "scenarioStudio.tsx"
    ).read_text(encoding="utf-8")
    assert "tCatalogWithFallback" not in source
    assert "catalog.localizationKeys.predicateHelp[definition.kind]" in source
    assert "return key ? tCatalog(key) : null" in source
    assert "t('workflow.research_scenario_catalog_help')" in source
    assert "{allDefinitions.length}" in source
    assert "!ortools && decisionKind !== 'play'" in source


def test_built_dashboard_is_arabic_first_and_served_from_the_api(root):
    from baloot_platform.service import api

    dashboard = root / "src" / "baloot_platform" / "service" / "static" / "foundry"
    index = (dashboard / "index.html").read_text(encoding="utf-8")
    source_root = (
        root / "src" / "baloot_platform" / "service" / "foundry_dashboard" / "src"
    )
    source_modules = _authored_dashboard_source_modules(source_root)
    source = _dashboard_source_corpus(source_modules)
    copy_profile = dict(api.ACTIVE_BUNDLE.component("product")["foundry_copy"])
    authored_product = yaml.safe_load(
        (root / "configs" / "product" / "studio_ortools.yaml").read_text(
            encoding="utf-8"
        )
    )
    configured_keys = {
        f"{group}.{key}"
        for group, messages in authored_product["foundry_copy"]["groups"].items()
        for key in messages
    }
    manifest = json.loads(
        (root / "generated" / "localization-usage-ortools-v4.json").read_text(
            encoding="utf-8"
        )
    )
    prefix = "localization.foundry_copy.groups."
    consumed_keys = {
        item["primitiveId"][len(prefix) :]
        for item in manifest["receipts"]
        if item["primitiveId"].startswith(prefix)
    }
    assert 'lang="ar"' in index
    assert 'dir="rtl"' in index
    assert "/foundry-assets/assets/" in index
    assert "{{foundry_copy_json}}" in index
    assert "{{foundry_page_title}}" in index
    assert re.search(r"[\u0600-\u06ff]", source) is None
    # Static UI calls, registered variants, and backend audit calls are parsed
    # into one executable manifest; spare governed copy is not permitted.
    assert consumed_keys == configured_keys
    assert len(configured_keys) >= 1000
    assert "/api/v2/sakka-generations" in source
    _assert_dashboard_source_order(
        source_modules,
        "if (current.status === 'completed') await refresh();",
        "setGenerationJob(current);",
    )
    assert "readFoundryCopy" in source
    assert "Foundry copy key is missing" in source
    assert "helpers.label_foundry_message" in source

    with TestClient(api.app) as client:
        response = client.get("/foundry")
        assert response.status_code == 200
        assert 'dir="rtl"' in response.text
        assert "{{foundry_copy_json}}" not in response.text
        assert copy_profile["page_title"] in response.text
        copy_response = client.get("/api/v1/foundry-copy")
        assert copy_response.status_code == 200
        assert copy_response.json()["copy"] == copy_profile
        assert copy_response.json()["locale"] == "ar-SA"
        assert copy_response.json()["direction"] == "rtl"
        favicon_response = client.get("/favicon.ico")
        assert favicon_response.status_code == 200
        assert favicon_response.headers["content-type"].startswith("image/svg+xml")
        assert (
            favicon_response.content
            == (
                root / "assets" / "video" / "professional" / "card-back.svg"
            ).read_bytes()
        )


def test_dashboard_connects_operator_workflow_and_governed_configuration(root):
    source_root = (
        root / "src" / "baloot_platform" / "service" / "foundry_dashboard" / "src"
    )
    source_modules = _authored_dashboard_source_modules(source_root)
    source = _dashboard_source_corpus(source_modules)
    source_by_module = dict(source_modules)
    research_api_source = source_by_module["api/research.ts"]
    research_model_source = source_by_module["features/research/model.ts"]
    research_scenario_source = source_by_module["features/research/scenarioStudio.tsx"]
    research_audit_source = source_by_module["features/research/audit.tsx"]
    research_audit_compact = re.sub(r"\s+", "", research_audit_source)
    research_error_copy_source = source_by_module["features/research/errorCopy.ts"]
    research_recipes_source = source_by_module["features/research/recipes.ts"]
    research_workspace_source = source_by_module["features/research/workspace.tsx"]
    main_source = source_by_module["main.tsx"]
    styles = (
        root
        / "src"
        / "baloot_platform"
        / "service"
        / "foundry_dashboard"
        / "src"
        / "styles.css"
    ).read_text(encoding="utf-8")
    vite_config = (
        root
        / "src"
        / "baloot_platform"
        / "service"
        / "foundry_dashboard"
        / "vite.config.ts"
    ).read_text(encoding="utf-8")
    product = yaml.safe_load(
        (root / "configs" / "product" / "studio_v1.yaml").read_text(encoding="utf-8")
    )

    copy = yaml.safe_load(
        (root / "configs" / "product" / "studio_ortools.yaml").read_text(encoding="utf-8")
    )["foundry_copy"]["groups"]["workflow"]
    assert copy["laffa_library_audit_only"] == "للمراجعة البحثية فقط؛ أهلية النشر غير مثبتة."
    assert copy["laffa_library_qualification_failures"] == "ملاحظات التأهيل والنشر"

    assert source.count('t("workflow.') >= 100
    for endpoint in (
        "/api/v2/campaigns",
        "/api/v2/analysis-batches",
        "/api/v2/finalist-validations",
        "/api/v2/render-batches",
        "/api/v2/video-production-batches",
        "/api/v2/sakkas",
        "/api/v2/publications",
        "/api/v2/laffat?",
        "/api/v2/laffat/",
        "/api/v1/configuration-registry/profiles",
        "/api/v1/configuration-registry/bundles",
    ):
        assert endpoint in source
    assert "publish:false" in source
    assert "'/api/v2/finalist-validations?limit=20'" in source
    assert "'/api/v2/render-batches?limit=20'" in source
    assert (
        "renderPage.items.find((item) => ['queued', 'running'].includes(item.status))"
        in source
    )
    assert "'/api/v2/render-batches?limit=100'" in source
    assert "-retry-${terminal.jobId}" in source
    assert "job.status === 'completed'" in source
    assert "['queued', 'running'].includes(job.status)" in source
    assert "const completedRenders = useMemo(" in source
    assert "<RecentRenderCard" in source
    assert 'className="video-library-copy"' in source
    assert 'className="video-library-actions"' in source
    _assert_dashboard_source_order(
        source_modules,
        "function RecentRenderCard",
        "workflow.video_library_open_source",
        "workflow.label_open_video",
    )
    assert "label_video_library" in source
    assert "batch_title" in source
    assert "batch_stage_qualifying" in source
    assert "batch_capacity_qualification" in source
    assert "scenarioCatalogReady" in source
    assert "proofBoundaryEvaluators" in research_scenario_source
    assert "catalogChoice(" in research_scenario_source
    assert "const previewSuitPrefix: Record<string, string>" in research_audit_source
    for suit, prefix in (
        ("spades", "S"),
        ("hearts", "H"),
        ("diamonds", "D"),
        ("clubs", "C"),
    ):
        assert f"{suit}: '{prefix}'" in research_audit_source
    assert (
        "['7', '8', '9', '10', 'J', 'Q', 'K', 'A'].includes(rank)"
        in research_audit_source
    )
    assert "resources.strategicAuction" not in research_workspace_source
    assert "resources.strategicPlay" not in research_workspace_source
    assert "setMessage((error as Error).message)" not in research_workspace_source
    assert "setMessage(error.message)" not in research_workspace_source
    assert "researchRecipeErrorMessage(error)" in research_workspace_source
    assert "researchRecipeSnapshot.error instanceof Error" not in main_source
    assert "researchRecipeErrorMessage(researchRecipeSnapshot.error)" in main_source
    assert "t(researchRecipeErrorCopyKey" not in source
    assert (
        "t('workflow.research_idempotency_unavailable')" in research_error_copy_source
    )
    assert "t('workflow.m_303534acbecf')" in research_error_copy_source
    assert "readonly code: ResearchRecipeErrorCode" in research_recipes_source
    assert "'workflow.research_idempotency_unavailable'" in research_recipes_source
    assert "'workflow.m_303534acbecf'" in research_recipes_source
    assert "new ResearchRecipeController(researchRecipeApi)" in main_source
    assert "const hydrateResearchRecipe = useCallback" in main_source
    assert "researchRecipeController.hydrateExact(code)" in main_source
    assert (
        "const recipe = await hydrateResearchRecipe(String(detail.recipeCode || ''))"
        in main_source
    )
    assert "hydrateResearchRecipe(item.recipeCode)" in main_source
    assert (
        "const retired = await researchRecipeController.retireRecipe(recipe.code)"
        in main_source
    )
    assert "[retired.code]:retired" in main_source
    assert "comparisonCampaigns" not in main_source
    assert "/api/v2/comparison-campaigns" not in main_source
    assert "ResearchComparisonWorkspace" not in main_source
    assert "listResearchRecipesApiV2ResearchRecipesGet" in research_api_source
    assert "getResearchRecipeApiV2ResearchRecipesCodeGet" in research_api_source
    assert "createResearchRecipeApiV2ResearchRecipesPost" in research_api_source
    assert (
        "retireResearchRecipeApiV2ResearchRecipesCodeRetirePost" in research_api_source
    )
    assert "headers: {'Idempotency-Key': idempotencyKey}" in research_api_source
    assert "nextCursor: page.nextCursor" in research_api_source
    assert "return api('/api/v2/research-recipes'" not in main_source
    assert "/api/v2/research-recipes?limit=100" not in source
    assert "researchRecipeController.loadMore()" in main_source
    assert "jack:'jack'" not in source and "king:'king'" not in source
    assert "function aggregateVideoBatchRejections" in source
    assert (
        "const rejectionSummaries = aggregateVideoBatchRejections(rejections)" in source
    )
    assert "rejectionSummaries.map" in source
    assert "finalistConcurrency" in source
    assert "videoBatchTarget" in source
    assert "videoProductionBatchProfile.sha256" in source
    assert "const videoBatchPreparedItems = useMemo" in source
    assert "function UnifiedLaffaLibrary" in source
    assert (
        'research_audit_trace:t("workflow.laffa_library_source_research_audit")'
        in source
    )
    assert 't("workflow.research_render_response_mismatch")' in source
    assert "function UnifiedLaffaDetails" in source
    assert "function QualifiedLaffaDetails" in source
    assert "function ResearchRunDetails" in source
    assert "cancelResearchRun(item)" in main_source
    assert 'item.status === \'queued\' || item.status === \'running\'' in main_source
    assert "scenarioStatus && scenarioYield" in source
    assert "scenarioYield.attempted" in source
    # The unified library owns result loading; background refresh must not
    # duplicate that read or silently replace a failed result page with [].
    assert "/api/v2/research-results?limit=" not in main_source
    assert "api<UnifiedLaffaPage>(`/api/v2/laffat?" in main_source
    assert "setResearchRuns(value.items)" in main_source
    assert "setResearchRuns((current) => current.map((item) => byCode.get(item.runCode) || item))" in main_source
    assert (
        "new EventSource(`/api/v2/research-runs/${encodeURIComponent(run.runCode)}/events`)"
        in research_audit_source
    )
    assert "source.addEventListener(type, onProgress)" in research_audit_source
    assert "if (!document.hidden) refresh()" in research_audit_source
    assert "function ResearchDisagreementAudit" in source
    assert 'id="all-generated-laffat"' not in source
    assert 'id="replay-valid-laffat"' not in source
    assert source.count('id="laffa-library"') == 1
    assert "<ResearchRenderPreview prepared={prepared}" in source
    assert (
        "export type ResearchRenderSourceProvenance = Readonly<"
        in research_model_source
    )
    assert "source: ResearchRenderSourceProvenance" in research_model_source
    assert "source?:ResearchRenderSourceProvenance" in source
    assert "prepared.source," in source
    assert "traceHash:String(prepared.source.traceHash||'')" not in source
    assert (
        "body:JSON.stringify({revealPolicy:'after_commit', exportPreset, observerSeat})"
        in source
    )
    assert "prepared.observerSeat!==requested" in source
    assert "source.observerSeat!==requested" in source
    assert "source.auditOnly!==!prepared.publicationEligible" in source
    assert "source.renderPurpose!==prepared.renderPurpose" in source
    assert "!Array.isArray(source.publicationBlockers)" in source
    assert "!source.publicationPolicy" in source
    assert (
        "onPrepare={(observerSeat,ordinal)=>prepareResearchVideo(detail, observerSeat, ordinal)}"
        in source
    )
    assert (
        "/api/v2/research-results/${detail.resultCode}/exact-state?observerSeat="
        in main_source
    )
    assert "onOpenExactState={async (observerSeat) =>" in main_source
    assert "research_scenario_observer_hand" in main_source
    assert "research_scenario_current_trick" in main_source
    assert "legalActions.map((action, index)" in main_source
    assert (
        "<JsonSummary value={(exactState.publicState || {}) as JsonObject}/>"
        not in main_source
    )
    assert "research-scenario-result" in research_audit_source
    assert "const isScenarioV4 = result.schemaVersion === 2" in research_audit_source
    assert "<ResearchWinningRootActionBoundary value={scenarioProof}/>" in (
        research_audit_source
    )
    assert "value.winningRootActionLowerBound ?? 0" in research_audit_source
    assert "value.winningRootActionUpperBound" in research_audit_source
    assert "onClick={() => onOpenExactState(observerSeat)}" in research_audit_source
    assert "!isScenarioV4 && <button" in research_audit_source
    assert "key={detail.resultCode}" in source
    assert "onClick={()=>onPrepare(observerSeat)}" in research_audit_compact
    assert (
        "onClick={()=>onPrepare(observerSeat,Number(item.ordinal))}"
        in research_audit_compact
    )
    assert (
        "constobserverSelectionReady=observerSeats.includes(observerSeat)"
        in research_audit_compact
    )
    assert "?projectedObserverSeats.filter(isSeatName):[];" in research_audit_compact
    assert (
        "disabled={!exactStateAvailable||!observerSelectionReady||!normalMediaAllowed}"
        in research_audit_compact
    )
    assert (
        "constactorView=observerSeat===String(item.actingSeat)"
        in research_audit_compact
    )
    assert (
        "constdecisionTruth=String(actorView?item.truthLabel:perspectiveRow?.truthLabel||'')"
        in research_audit_compact
    )
    assert (
        "{actorView&&<div><span>{t('workflow.research_recommended_action')}"
        in research_audit_compact
    )
    assert "{actorView&&!isNativeV3&&alternatives.length>0&&" in research_audit_compact
    assert (
        "{actorView&&!isNativeV3&&<divclassName=\"wide\"><span>{t('workflow.research_selection_reason')}"
        in research_audit_compact
    )
    _assert_dashboard_source_order(
        source_modules,
        'actorView ? <>{!isNativeV3 && <><div className="information-set-proof"',
        "{legalActions.length > 0 &&",
        '</> : <><p className="drawer-lead audit-only">',
    )
    assert (
        "{actorView&&!isNativeV3&&<divclassName=\"wide\"><span>{t('workflow.research_analysis_reason')}"
        in research_audit_compact
    )
    assert "{actorView&&continuation.length>0&&" in research_audit_compact
    assert "{actorView&&Array.isArray(item.tacticalEvidence)" in research_audit_compact
    assert (
        "constpreviewSeatRing:SeatName[]=['north','west','south','east'];"
        in research_audit_compact
    )
    assert "functionpreviewActualAtVisual" in research_audit_compact
    assert "functionpreviewVisualOfActual" in research_audit_compact
    assert "functionpreviewRelativePair" in research_audit_compact
    assert (
        "previewSeatRing[(previewSeatRing.indexOf(visual)+observerIndex-2+previewSeatRing.length)%previewSeatRing.length]"
        in research_audit_compact
    )
    assert (
        "previewSeatRing[(previewSeatRing.indexOf(actual)-observerIndex+2+previewSeatRing.length)%previewSeatRing.length]"
        in research_audit_compact
    )
    assert (
        "constobserverTeamIndex=observer==='north'||observer==='south'?0:1;"
        in research_audit_compact
    )
    assert "constscore=previewRelativePair(" in research_audit_compact
    assert (
        "constactualSeat=previewActualAtVisual(visualSeat,observer)"
        in research_audit_compact
    )
    assert (
        "constvisualSeat=previewVisualOfActual(actualSeat,observer)"
        in research_audit_compact
    )
    assert (
        "prepared.publicationEligible?t('workflow.laffa_library_content_candidate')"
        in research_audit_compact
    )
    assert (
        "constsourceKind=String(prepared.source.originKind||prepared.source.kind||'');"
        in research_audit_compact
    )
    assert "typeofresult.publicationEligible==='boolean'" in research_audit_compact
    assert "{publicationEligibilityKnown&&<section" in research_audit_compact
    assert (
        "laffaQualificationLabel[prepared.qualificationStatus]"
        in research_audit_compact
    )
    assert "prepared.source.attemptStatus&&<Stat" in research_audit_compact
    assert "t('workflow.research_filter_observer_seat')" in research_audit_compact
    assert "laffaSourceLabel[sourceKind]||sourceKind" in research_audit_compact
    assert 'laffa:t("workflow.laffa_library_source_archive")' in source
    assert "prepared.source.publicationBlockers" in source
    assert (
        "source?.auditOnly === false && source?.publicationEligible === true && source?.renderPurpose === 'content_candidate'"
        in source
    )
    assert "const publishable=item.publicationEligible===true" in source
    assert (
        "item.qualificationStatus==='qualified'?t(\"workflow.batch_render_selected\")"
        not in source
    )
    assert "const publishable=item.qualificationStatus==='qualified'" not in source
    assert "function renderJobReferencesUnifiedLaffa" in source
    assert (
        "source.kind === 'archived_game' || source.kind === 'sakka_round' || source.kind === 'research_audit_trace' ? 'laffa'"
        in source
    )
    assert "section=laffat&laffa=" in source
    assert "#laffa-library" in source
    assert "api<UnifiedLaffaPage>(`/api/v2/laffat?" in source
    assert "const unifiedLaffaSourceRevision = useMemo" in source
    assert "setRenderHistory(page.items)" in source
    assert (
        "renderHistory.filter((item) => ['queued', 'running'].includes(item.status))"
        in source
    )
    assert "setDrawer((current) => current?.kind === 'render'" in source
    assert "`${item.runCode}:${item.status}:${item.resultCount}`" in source
    assert "[catalogs, renderHistory, researchRuns, sakkas]" in main_source
    assert "setResearchResults" not in main_source
    assert "`${item.jobId}:${item.status}`" in source
    assert "if (view !== 'laffat') return;" in source
    assert (
        "[laffaFilters, loadUnifiedLaffat, unifiedLaffaSourceRevision, view]" in source
    )
    assert (
        "sourceKind" in source
        and "qualificationStatus" in source
        and "renderStatus" in source
    )
    assert "prepareUnifiedLaffaVideo" in source
    assert (
        "const preparedQualified = item.sourceLocator.kind === 'qualified_laffa'"
        in source
    )
    assert "const sourceCode = item.sourceLocator.code" in source
    assert "archived?.code || research?.code" not in source
    assert "`/api/v2/laffat/${encodeURIComponent(sourceCode)}:prepare-render`" in source
    assert "<ResearchRenderPreview prepared={prepared}" in source
    assert "researchFailureLabel" in source
    assert "minimumSurroundingQuality" in source
    assert "maximumRegret" in source
    assert "ScenarioStudio" in source
    assert "codeSplitting" in vite_config
    assert "name: 'react'" in vite_config and "name: 'tanstack'" in vite_config
    assert "capabilities.research.recipeConfiguration" in source
    assert "capabilities.research.scenarioConfiguration" in source
    assert "authoringMode" not in research_workspace_source
    assert "ScenarioStudio" in research_workspace_source
    assert "configuration.outputPolicy.resultCount" in research_scenario_source
    assert "configuration.outputPolicy.videoCount" in research_scenario_source
    assert (
        "recipeConfiguration={capabilities.research.recipeConfiguration}"
        in research_workspace_source
    )
    assert "refresh={props.refresh}" in research_workspace_source
    assert "await refresh();" in research_scenario_source
    assert (
        "/api/v2/research-runs/${encodeURIComponent(run.runCode)}"
        in research_scenario_source
    )
    assert "const [current, stats, recipe] = await Promise.all" in main_source
    assert "detail.classification === 'scenario_v4' ? 1 : undefined" not in main_source
    assert "configuration.predicateRegistry.definitions" in research_scenario_source
    assert "tCatalog(definition.messageArKey)" in research_scenario_source
    assert "tCatalog(definition.messageEnKey)" in research_scenario_source
    assert "schema['x-messageArKey']" in research_scenario_source
    assert "schema?.['x-enumMessageArKeys']" in research_scenario_source
    assert "scenario_catalog_argument_label_missing" in research_scenario_source
    assert "availableClaims.map" in research_scenario_source
    assert "authoringCatalog.proof.utilities.map" in research_scenario_source
    assert "configuration.proofUtilities.map" not in research_scenario_source
    assert "availableLineUniverses.map" in research_scenario_source
    assert "authoringCatalog.proof.horizonUnits.map" in research_scenario_source
    assert "configuration.proofHorizonUnits.map" not in research_scenario_source
    assert "Object.keys(claimLabels).map" not in research_scenario_source
    assert "humanKey('declarationOrder')" not in research_scenario_source
    assert "humanKey('state')" not in research_scenario_source
    assert "humanKey('awarded_point_margin')" not in research_scenario_source
    assert "humanKey('all_legal')" not in research_scenario_source
    assert "optionLabel(catalog,item.kind)" in research_scenario_source
    assert "optionLabel(catalog,String(item.value))" in research_scenario_source
    assert "registeredLabel(scenarioOptionLabels" not in research_scenario_source
    assert "scenarioProof.rootActions" in research_audit_source
    assert "research_scenario_root_actions" in research_audit_source
    assert "research_scenario_action_wins" in research_audit_source
    assert "research_scenario_action_loses" in research_audit_source
    assert "<JsonSummary value={scenarioProof}/>" not in research_audit_source
    assert "scenario_guided_actor_safe" in research_scenario_source
    assert "research_scenario_decision_action_index" in research_scenario_source
    assert "actionIndex: decisionActionIndex" in research_scenario_source
    assert "research_scenario_decision_remaining_cards" in research_scenario_source
    assert "remainingCards: decisionRemainingCards" in research_scenario_source
    assert "actingSeat: decisionSeat" in research_scenario_source
    assert "exact_trace_prefix" in research_scenario_source
    assert "supportedDecisions.length !== 4" in research_scenario_source
    assert "item.searchGenerationSupported" in research_scenario_source
    assert "selectedDecisionCapability?.proofModes" in research_scenario_source
    assert "decisionKind === 'escalation'" in research_scenario_source
    assert "decisionKind === 'project'" in research_scenario_source
    assert "projectIndex" in research_scenario_source
    assert "catalog.logicalOperators" in research_scenario_source
    assert "configuration.logicalOperators" not in research_scenario_source
    assert ".map((operator)" in research_scenario_source
    assert "validation?.normalizedExactState" in research_scenario_source
    assert "ExactTraceEditor" in research_scenario_source
    assert "ExactActionEditor" in research_scenario_source
    assert "PreviewCardFace" in research_scenario_source
    assert "schemaHasVisualControl" in research_scenario_source
    assert "scenario-deck-editor" in research_scenario_source
    assert "scenario-action-timeline" in research_scenario_source
    assert "scenario-structured-list" in research_scenario_source
    assert "scenario-card-picker" in research_scenario_source
    assert research_scenario_source.count(
        'className="scenario-structured-field scenario-card-field"'
    ) == 2
    assert ".scenario-card-field{grid-column:1/-1}" in styles
    assert "deck.length !== 32" in research_scenario_source
    assert "previousIndex = deck.indexOf(nextCard)" in research_scenario_source
    assert "highRank: selected.highRankId" in research_scenario_source
    assert "{item.highRankId} · {optionLabel(catalog,item.highRank)}" in research_scenario_source
    assert "options.ranks.indexOf(selected.highRank)" not in research_scenario_source
    assert "Number.isInteger(item.highRankId)" in research_scenario_source
    assert "research_scenario_exact_trace_advanced" in research_scenario_source
    assert '<textarea dir="ltr" value={raw}' not in research_scenario_source
    assert "JSON.stringify(item, null, 2)" not in research_scenario_source
    assert (
        "schema.type === 'array' || schema.type === 'object'"
        not in research_scenario_source
    )
    assert "/api/v2/research-recipes:validate" in research_scenario_source
    assert "/api/v2/research-recipes:estimate" in research_scenario_source
    assert "if (!validation?.valid || validatedFingerprint !== submittedFingerprint)" in research_scenario_source
    assert "round_state" not in research_scenario_source.lower()
    assert "setTacticMinimumScore(70)" not in source
    assert "titleAr:tacticTitle" not in source
    assert "/api/v2/configuration/bundles?status=published&limit=200" in source
    assert (
        "/api/v2/domain-capabilities?bundleRevisionId=${encodeURIComponent(selectedBundleRevisionId)}"
        in source
    )
    assert "const sequence=++requestSequence.current" in source
    assert "return()=>controller.abort()" in source
    assert "configurationSelection" in source
    assert "String(authority.bundleRevisionId ?? '') !== selectedBundle.revisionId" in research_scenario_source
    assert "schemaVersion: 4" in research_scenario_source
    assert "configuration: {schemaVersion: 1, bundleRevisionId: selectedBundle.revisionId}" in research_scenario_source
    assert "scenario: {" in research_scenario_source
    assert "output: {kind: outputKind, resultCount: desiredResultCount}" in research_scenario_source
    assert "domainHash:capabilities.sha256" not in source
    assert "function ResearchConfigurationDetails" in source
    assert "/api/v2/configuration/profiles?limit=1000" in source
    assert "profile?.version||bundle.version" not in source
    assert "workflow.research_review_configuration" in source
    assert ".research-bundle-selector" in styles
    assert "item.bundleId === runtimeBundle.bundleId" in source
    assert "Number(right.sequence||0)-Number(left.sequence||0)" in source
    assert "Object.entries(item.componentRevisions).every" in source
    assert "profile.revisionId === revisionId && profile.status === 'published'" in source
    assert "/api/v1/configuration-registry/profiles?limit=1000" in source
    assert ".research-card-picker" in styles
    assert "research-qualification-summary" in source
    assert "function ResearchPerformanceSummary" in source
    assert "hasPerformanceEvidence=value.shardCount>0" in source
    assert "performanceByRun[value.runCode]?.performance" not in source
    assert "/api/v2/research-runs/${code}/stats" not in source
    assert "/api/v2/research-runs/${item.runCode}/stats" in source
    assert "research_metric_unavailable" in source
    assert "costProxy.wallSecondsPerQualified" in source
    assert "solverStatusDistribution" in source
    assert "operations.retryCount===null" in source
    assert "onClick={onRuns}" in source and "onClick={onResults}" in source
    assert "sequentialSearch: ResearchSequentialSearchSummary" in research_model_source
    for receipt_counter in (
        "admittedSeedCount",
        "exactRejectionCount",
        "noGoodAppliedCount",
        "noGoodUnappliedCount",
        "continuationSolveCount",
        "continuationStatusDistribution",
        "stopReasonDistribution",
    ):
        assert receipt_counter in research_model_source
        assert receipt_counter in research_audit_source
    assert "ResearchSequentialSearchSummary = Readonly<{" in research_model_source
    assert "admitted: true;" in research_model_source
    assert "receiptBundle: Readonly<JsonObject>;" in research_model_source
    assert "sequential-search-receipts?offset=0&limit=25" in main_source
    assert "receiptPage.items.map" in research_audit_source
    assert "receiptPage.items.sort" not in research_audit_source
    assert "if (receiptState !== 'idle') return;" in research_audit_source
    assert "if (event.currentTarget.open) loadReceipts();" in research_audit_source
    assert "JSON.stringify(item.receiptBundle,null,2)" in research_audit_compact
    assert "rejectionFeedbackRows" not in research_audit_source
    assert ".research-sequential-summary" in styles
    assert ".research-sequential-receipt-list" in styles
    assert ".research-sequential-identities" in styles
    sequential_receipt_copy = {
        key
        for key in product["foundry_copy"]["groups"]["workflow"]
        if key.startswith("research_sequential_")
        and not key.startswith("research_sequential_response_")
    }
    consumed_sequential_receipt_copy = set(
        re.findall(
            r"t\(['\"]workflow\.(research_sequential_[^'\"]+)['\"]\)",
            research_audit_source,
        )
    ) - {"research_sequential_response_status"}
    assert consumed_sequential_receipt_copy == sequential_receipt_copy
    assert "optimalityCutCount" not in source
    assert "onOpenDisagreement(item)" in source
    assert (
        "request<ResearchResultPage>(`/api/v2/research-results?${query.toString()}`)"
        in research_scenario_source
    )
    configuration = research_recipe_configuration_v1()
    assert configuration["recipeSchemaVersion"] == 3
    assert configuration["supportedRecipeSchemaVersions"] == [1, 2, 3]
    assert configuration["recipePresets"]
    assert configuration["resourcePresets"]
    assert all(
        item.get("labelKey")
        and item.get("helpKey")
        and item.get("exampleKey")
        and item.get("example")
        for item in configuration["relationships"]
    )
    missing_operator_fields = sorted(
        path
        for path, definition in configuration["fields"].items()
        if definition.get("mode") == "operator" and path not in source
    )
    assert missing_operator_fields == []
    assert "Object.entries(resultFilters).forEach" in research_scenario_source
    for query_parameter, facet_name in (
        ("scenarioTemplate", "scenario_template"),
        ("scenarioOrigin", "scenario_origin"),
        ("scenarioStatus", "scenario_status"),
        ("requestedProofMode", "requested_proof_mode"),
        ("achievedProofMode", "achieved_proof_mode"),
        ("proofCompletion", "proof_completion"),
        ("proofVisibility", "proof_visibility"),
        ("targetTeam", "target_team"),
        ("contractOutcome", "contract_outcome"),
        ("projectKind", "project_kind"),
        ("projectValue", "project_value"),
        ("projectState", "project_state"),
        ("winningRootActionBand", "winning_root_action_band"),
        ("exactStateAvailable", "exact_state_available"),
        ("decisionKind", "decision_kind"),
        ("researchBackend", "research_backend"),
        ("qualificationAuthority", "qualification_authority"),
        ("runtimeCompatibilityStatus", "runtime_compatibility_status"),
        ("configurationBundle", "configuration_bundle"),
        ("exactStateId", "exact_state_id"),
        ("scenarioCategory", "scenario_category"),
        ("variantCollection", "variant_collection"),
    ):
        assert f"{query_parameter}:'{facet_name}'" in research_scenario_source
    assert "setStudioResults(page.items)" in research_scenario_source
    assert "getResearchResultOperatorAudit(result.resultCode)" in research_audit_source
    assert "setStudioResultTotal(page.total)" in research_scenario_source
    assert "setResultGroups(page.groups)" in research_scenario_source
    assert "query.set('groupBy',resultGroupBy)" in research_scenario_source
    assert "sort: 'score_desc'" in research_scenario_source
    assert "useState<ResearchV4ResultFilters>({})" in research_scenario_source
    assert "setResultGroupBy('')" in research_scenario_source
    assert 'className="research-server-groups"' in research_scenario_source
    assert ".research-server-groups" in styles
    assert ".research-evidence-compare" in styles
    assert "laffa_library_column_decision_score" in source
    assert "laffa_library_column_laffa_score" in source
    assert "renderJobReferencesBundle" in source
    assert "useTable({" in source
    assert "batch.result?.preparedItems" not in source
    assert "const batchHistoryFloor = videoBatchHistory.reduce" in source
    assert (
        "Math.max(waveSeedCount, Number(item.result?.roundsGenerated || 0))" in source
    )
    assert "const requestedSeed = Math.max(campaignFloor, batchHistoryFloor)" in source
    assert "batch_advanced_title" in source
    assert ".video-batch-layout" in styles
    assert ".data-table-wrap" in styles
    assert ".library-toolbar" in styles
    assert styles.index(".unified-laffa-toolbar{") < styles.index(
        "@media(max-width:900px){.library-toolbar{"
    )
    assert "campaigns/${campaign.campaignHash}:execute" in source
    assert "analysis-batches/${job.jobId}:execute" in source
    assert "exportPreset:'hooked_flagship'" in source
    assert "<RenderOutputCard job={renderJob} />" in source
    assert '<video controls playsInline preload="metadata"' in source
    assert 'href="/studio"' not in source
    assert "date.format(new Date(item.updatedAt))" in source
    assert "foundry-finalist-${analysisJob.jobId}-${candidate.candidateHash}" in source
    assert "readiness(right) - readiness(left)" in source
    assert "Date.parse(right.createdAt) - Date.parse(left.createdAt)" in source
    assert "item.campaignHash === current.campaignHash" in source
    assert "drawer.kind !== 'campaign'" in source
    assert (
        "<CampaignDetails campaign={selected} shards={shards} attempts={attempts} />"
        in source
    )
    assert "selectedCandidateCount === 0" in source
    assert (
        "setAnalysisJob(job); setCandidatePage(null); setFinalistJob(null);" in source
    )
    assert "JsonEditor" in source
    assert "function ConfigObjectSection" in source
    assert "expanded && <JsonEditor" in source
    assert "/api/v2/configuration/profile-schemas/" in source
    assert "method:'PATCH'" in source
    assert "expectedEtag:selected.etag" in source
    assert ":validate`" in source
    assert ":compile`" in source
    assert "allowDraftComponents:selected.status==='draft'" in source
    assert "!profileCompilation?.publishable" in source
    assert "function RawJsonFallback" in source
    assert "if (Array.isArray(schema?.enum)) return schema.enum;" in source_by_module["features/configuration/model.ts"]
    assert "const options = configurationEditorEnum(schema);\n  if (options?.length) {" in main_source
    assert "schema?.minimum" in source
    assert ":rollback`" in source
    assert "rules_revision_picker" in source
    assert ".draft-proof-flow" in styles
    assert ".raw-json-fallback" in styles
    assert "const thrillFieldAr" in source
    assert "path.includes('thrill_profile')" in source
    assert "configFieldName(key,child,nextPath)" in source
    assert "return contextualKeyName(key, path)" in main_source
    assert 't("helpers.thrill_profile_example")' in source
    assert 'role="switch"' in source
    assert 'aria-modal="true"' in source
    assert "prefers-reduced-motion" in styles
    assert ".info-wrap:focus-within" in styles
    assert "@media(max-width:520px)" in styles


def test_api_documentation_is_offline_and_searchable(root):
    from baloot_platform.service import api

    source = (
        root / "src" / "baloot_platform" / "service" / "static" / "api_docs.html"
    ).read_text(encoding="utf-8")
    assert "cdn.jsdelivr.net" not in source
    assert "fetch('/openapi.json'" in source
    assert 'aria-label="Search API endpoints"' in source

    with TestClient(api.app) as client:
        for path in ("/docs", "/redoc"):
            response = client.get(path)
            assert response.status_code == 200
            assert "BalootX API Reference" in response.text


_NPM_VERSION_NUMBER = r"(?:0|[1-9][0-9]*)"
_NPM_PRERELEASE_ID = rf"(?:{_NPM_VERSION_NUMBER}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_EXACT_NPM_VERSION = re.compile(
    rf"{_NPM_VERSION_NUMBER}\.{_NPM_VERSION_NUMBER}\.{_NPM_VERSION_NUMBER}"
    rf"(?:-{_NPM_PRERELEASE_ID}(?:\.{_NPM_PRERELEASE_ID})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def _assert_dashboard_npm_pins(manifest, lock):
    assert lock["lockfileVersion"] == 3
    packages = lock["packages"]
    for section in ("dependencies", "devDependencies"):
        pins = manifest[section]
        assert pins == packages[""][section], f"lockfile root differs for {section}"
        for name, spec in pins.items():
            locked = packages[f"node_modules/{name}"]
            version = spec
            if spec.startswith("npm:"):
                target, _, version = spec.removeprefix("npm:").rpartition("@")
                assert target and locked.get("name") == target, (
                    f"wrong alias target for {name}"
                )
            else:
                assert locked.get("name", name) == name, f"wrong package for {name}"
            assert _EXACT_NPM_VERSION.fullmatch(version), (
                f"{name} needs an exact version"
            )
            assert locked["version"] == version, f"locked version differs for {name}"

    for name, version in manifest["overrides"].items():
        assert _EXACT_NPM_VERSION.fullmatch(version), f"{name} needs an exact override"
        package_path = f"node_modules/{name}"
        locked_versions = {
            entry["version"]
            for path, entry in packages.items()
            if path == package_path or path.endswith(f"/{package_path}")
        }
        assert locked_versions == {version}, (
            f"locked versions differ for override {name}"
        )


@pytest.mark.parametrize(
    ("spec", "root_spec", "locked_version", "locked_name", "error"),
    [
        ("2.3.5", "2.3.5", "2.3.5", "tool", None),
        ("*", "*", "2.3.5", "tool", "needs an exact version"),
        ("^2.3.5", "^2.3.5", "2.3.5", "tool", "needs an exact version"),
        ("2.3.5", "2.3.4", "2.3.5", "tool", "lockfile root differs"),
        ("2.3.5", "2.3.5", "2.3.4", "tool", "locked version differs"),
        (
            "npm:@scope/tool@2.3.5",
            "npm:@scope/tool@2.3.5",
            "2.3.5",
            "@scope/tool",
            None,
        ),
        (
            "npm:other@~2.3.5",
            "npm:other@~2.3.5",
            "2.3.5",
            "other",
            "needs an exact version",
        ),
        ("npm:other@2.3.5", "npm:other@2.3.5", "2.3.5", "tool", "wrong alias target"),
    ],
)
def test_dashboard_npm_pin_checks(spec, root_spec, locked_version, locked_name, error):
    manifest = {"dependencies": {"tool": spec}, "devDependencies": {}, "overrides": {}}
    lock = {
        "lockfileVersion": 3,
        "packages": {
            "": {"dependencies": {"tool": root_spec}, "devDependencies": {}},
            "node_modules/tool": {"name": locked_name, "version": locked_version},
        },
    }
    if error is None:
        _assert_dashboard_npm_pins(manifest, lock)
    else:
        with pytest.raises(AssertionError, match=error):
            _assert_dashboard_npm_pins(manifest, lock)


def test_dashboard_build_and_wheel_inputs_are_exactly_pinned(root):
    dashboard = root / "src" / "baloot_platform" / "service" / "foundry_dashboard"
    package = (dashboard / "package.json").read_text(encoding="utf-8")
    manifest = json.loads(package)
    lock = json.loads((dashboard / "package-lock.json").read_text(encoding="utf-8"))
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert {"react", "@tanstack/react-table"} <= manifest["dependencies"].keys()
    dev = manifest["devDependencies"]
    assert {"vite", "@babel/parser", "@hey-api/openapi-ts", "vitest"} <= dev.keys()
    assert dev["@typescript/native"].startswith("npm:typescript@")
    assert dev["typescript"].startswith("npm:@typescript/typescript6@")
    assert {"js-yaml", "nanoid"} <= manifest["overrides"].keys()
    _assert_dashboard_npm_pins(manifest, lock)
    assert (
        '"generate:api": "openapi-ts --file openapi-ts.config.mjs --no-log-file"'
        in package
    )
    assert '"check:api-generated": "node scripts/check-generated-api.mjs"' in package
    assert (
        '"test:research": "vitest --root ../../../.. run tests/frontend/research_recipes.spec.ts tests/frontend/card_selector.spec.tsx tests/frontend/ortools_scenario_ui.spec.ts tests/frontend/ortools_scenario_predicate_capabilities.spec.ts tests/frontend/video_playback_state.spec.ts tests/frontend/video_qr_policy.spec.ts"'
        in package
    )
    assert '"openapi-fetch"' not in package
    assert '"static/foundry/*.html"' in pyproject
    assert '"static/foundry/assets/*.css"' in pyproject
    assert '"static/foundry/assets/*.js"' in pyproject
