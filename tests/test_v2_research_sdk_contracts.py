from __future__ import annotations

import json
from pathlib import Path

import httpx

import sdk.python.client as python_sdk
from sdk.python.client import BalootClient


ROOT = Path(__file__).resolve().parents[1]


def test_python_sdk_maps_the_research_and_unified_laffa_contracts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='id: 1\nevent: completed\ndata: {"type":"completed"}\n\n',
            )
        return httpx.Response(200, json={"ok": True})

    client = BalootClient(base_url="http://sdk.test")
    client.client.close()
    client.client = httpx.Client(
        base_url="http://sdk.test", transport=httpx.MockTransport(handler)
    )
    try:
        payload = {"schemaVersion": 2}
        client.domain_capabilities(include_payloads=True)
        client.domain_capabilities(bundle_revision_id="CBR-SDK-A")
        client.domain_compilation("a" * 64, include_payloads=True)
        client.domain_compilation_coverage("a" * 64)
        client.rule_primitives()
        client.rule_primitives(bundle_revision_id="CBR-SDK-B")
        client.validate_research_recipe(payload)
        client.estimate_research_recipe(payload)
        client.create_video(
            bundleRevisionId="CBR-SDK-VIDEO",
            idempotency_key="legacy-video-key",
        )
        client.create_research_recipe(payload)
        client.research_recipes(status="published", cursor="recipe-cursor", limit=25)
        client.research_recipe("RCP-ABC123")
        client.retire_research_recipe("RCP-ABC123")
        client.create_research_run("RCP-ABC123", idempotency_key="run-key")
        client.research_runs(status="completed", limit=25)
        client.research_run("RUN-ABC123")
        client.research_run_stats("RUN-ABC123")
        client.research_stats()
        client.cancel_research_run("RUN-ABC123")
        assert list(client.research_run_events("RUN-ABC123")) == [
            {"type": "completed"}
        ]
        client.create_comparison_campaign("RCP-ABC123", idempotency_key="cmp-key")
        client.comparison_campaigns(limit=25)
        client.comparison_campaign("CMP-ABC123")
        assert list(client.comparison_campaign_events("CMP-ABC123")) == [
            {"type": "completed"}
        ]
        client.research_results(
            tactical_tag="only_winning_move",
            minimum_score=80,
            scenario_template="SCN-SDK",
            scenario_origin="exact_trace_prefix",
            scenario_status="satisfied",
            requested_proof_mode="exact",
            achieved_proof_mode="exact",
            proof_completion="exact_complete",
            proof_visibility="complete_state",
            target_team="north_south",
            contract_outcome="lost",
            project_kind="hundred_sequence",
            project_value="100",
            project_state="awarded",
            winning_root_action_band="1",
            exact_state_available=True,
            decision_kind="play",
            research_backend="ortools_cp_sat_v2",
            qualification_authority="cp_sat",
            runtime_compatibility_status="passed",
            configuration_bundle="CBR-SDK-V4",
            exact_state_id="STATE-1234567890ABCDEF1234",
            scenario_category="proof",
            variant_collection="RUN-ABCDEFGHIJKL",
            group_by="scenario_category",
            limit=25,
        )
        client.research_result("LAF-ABC123")
        client.research_result_decisions("LAF-ABC123")
        client.research_result_perspectives("LAF-ABC123", "south")
        client.research_result_decision("LAF-ABC123", 7)
        client.prepare_research_result_render(
            "LAF-ABC123", exportPreset="hooked_flagship", observerSeat="west"
        )
        client.prepare_research_decision_render(
            "LAF-ABC123", 7, exportPreset="decision_clip", observerSeat="east"
        )
        client.research_result_renders("LAF-ABC123")
        client.laffat(
            source_kind="archived_game", replay_valid=True, render_status="ready"
        )
        client.laffa("GAME-ABC123")
        client.prepare_laffa_render(
            "GAME-ABC123", exportPreset="hooked_flagship", observerSeat="north"
        )
        client.laffa_renders("GAME-ABC123")
    finally:
        client.close()

    by_path = {request.url.path: request for request in requests}
    by_method_path = {(request.method, request.url.path): request for request in requests}
    domain_requests = [
        request
        for request in requests
        if request.url.path == "/api/v2/domain-capabilities"
    ]
    assert domain_requests[0].url.params["includePayloads"] == "true"
    assert "bundleRevisionId" not in domain_requests[0].url.params
    assert domain_requests[1].url.params["bundleRevisionId"] == "CBR-SDK-A"
    primitive_requests = [
        request
        for request in requests
        if request.url.path == "/api/v2/rule-primitives"
    ]
    assert "bundleRevisionId" not in primitive_requests[0].url.params
    assert primitive_requests[1].url.params["bundleRevisionId"] == "CBR-SDK-B"
    assert by_path["/api/v2/research-recipes"].url.params["status"] == "published"
    assert by_path["/api/v2/research-recipes"].url.params["cursor"] == "recipe-cursor"
    assert by_method_path[("POST", "/api/v2/research-runs")].headers["Idempotency-Key"] == "run-key"
    assert json.loads(by_method_path[("POST", "/api/v2/research-runs")].content) == {
        "recipeCode": "RCP-ABC123"
    }
    assert by_method_path[("POST", "/api/v2/comparison-campaigns")].headers["Idempotency-Key"] == "cmp-key"
    legacy_video = by_method_path[("POST", "/api/v1/video-jobs")]
    assert legacy_video.headers["Idempotency-Key"] == "legacy-video-key"
    assert json.loads(legacy_video.content) == {
        "bundleRevisionId": "CBR-SDK-VIDEO"
    }
    assert by_path["/api/v2/research-results"].url.params["tacticalTag"] == "only_winning_move"
    assert by_path["/api/v2/research-results"].url.params["minimumScore"] == "80"
    assert dict(by_path["/api/v2/research-results"].url.params).items() >= {
        "scenarioTemplate": "SCN-SDK",
        "scenarioOrigin": "exact_trace_prefix",
        "scenarioStatus": "satisfied",
        "requestedProofMode": "exact",
        "achievedProofMode": "exact",
        "proofCompletion": "exact_complete",
        "proofVisibility": "complete_state",
        "targetTeam": "north_south",
        "contractOutcome": "lost",
        "projectKind": "hundred_sequence",
        "projectValue": "100",
        "projectState": "awarded",
        "winningRootActionBand": "1",
        "exactStateAvailable": "true",
        "decisionKind": "play",
        "researchBackend": "ortools_cp_sat_v2",
        "qualificationAuthority": "cp_sat",
        "runtimeCompatibilityStatus": "passed",
        "configurationBundle": "CBR-SDK-V4",
        "exactStateId": "STATE-1234567890ABCDEF1234",
        "scenarioCategory": "proof",
        "variantCollection": "RUN-ABCDEFGHIJKL",
        "groupBy": "scenario_category",
    }.items()
    assert by_path["/api/v2/research-results/LAF-ABC123/perspectives"].url.params[
        "observerSeat"
    ] == "south"
    assert by_path["/api/v2/laffat"].url.params["sourceKind"] == "archived_game"
    assert by_path["/api/v2/laffat"].url.params["replayValid"] == "true"
    assert by_path["/api/v2/laffat"].url.params["renderStatus"] == "ready"
    assert json.loads(
        by_path["/api/v2/research-results/LAF-ABC123:prepare-render"].content
    ) == {"exportPreset": "hooked_flagship", "observerSeat": "west"}
    assert json.loads(
        by_path[
            "/api/v2/research-results/LAF-ABC123/decisions/7:prepare-render"
        ].content
    ) == {"exportPreset": "decision_clip", "observerSeat": "east"}
    assert json.loads(
        by_path["/api/v2/laffat/GAME-ABC123:prepare-render"].content
    ) == {"exportPreset": "hooked_flagship", "observerSeat": "north"}


def test_python_sdk_recipe_mutations_and_pagination_preserve_compatibility(
    monkeypatch,
) -> None:
    requests: list[httpx.Request] = []
    generated: list[str] = []

    def fake_uuid4() -> str:
        value = f"generated-{len(generated) + 1}"
        generated.append(value)
        return value

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"items": [{"schemaVersion": 3}], "nextCursor": None},
            )
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(python_sdk, "uuid4", fake_uuid4)
    client = BalootClient(base_url="http://sdk.test")
    client.client.close()
    client.client = httpx.Client(
        base_url="http://sdk.test", transport=httpx.MockTransport(handler)
    )
    try:
        payload = {"schemaVersion": 3}
        client.create_research_recipe(payload)
        client.create_research_recipe(payload, idempotency_key="")
        client.retire_research_recipe("RCP-ABC123")
        client.retire_research_recipe("RCP-ABC123", idempotency_key="retire-key")
        client.retire_research_recipe("RCP-ABC123", idempotency_key="")
        legacy_page = client.research_recipes(status="published", limit=25)
        blank_page = client.research_recipes(cursor="", limit=25)
        cursor_page = client.research_recipes(cursor="NEXT", limit=25)
    finally:
        client.close()

    assert generated == ["generated-1", "generated-2"]
    assert [request.headers["Idempotency-Key"] for request in requests[:5]] == [
        "generated-1",
        "",
        "generated-2",
        "retire-key",
        "",
    ]
    assert "cursor" not in requests[5].url.params
    assert requests[6].url.params.get_list("cursor") == [""]
    assert requests[7].url.params["cursor"] == "NEXT"
    assert legacy_page == {"items": [{"schemaVersion": 3}], "nextCursor": None}
    assert blank_page == legacy_page
    assert cursor_page == legacy_page


def test_all_sdks_expose_the_same_research_and_laffa_surface() -> None:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(
        encoding="utf-8"
    )
    go_source = (ROOT / "sdk/go/client.go").read_text(encoding="utf-8")

    method_names = (
        ("domain_capabilities", "domainCapabilities", "DomainCapabilities"),
        ("domain_compilation", "domainCompilation", "DomainCompilation"),
        ("domain_compilation_coverage", "domainCompilationCoverage", "DomainCompilationCoverage"),
        ("rule_primitives", "rulePrimitives", "RulePrimitives"),
        ("validate_research_recipe", "validateResearchRecipe", "ValidateResearchRecipe"),
        ("estimate_research_recipe", "estimateResearchRecipe", "EstimateResearchRecipe"),
        ("create_research_recipe", "createResearchRecipe", "CreateResearchRecipe"),
        ("research_recipes", "researchRecipes", "ResearchRecipes"),
        ("research_recipe", "researchRecipe", "ResearchRecipe"),
        ("retire_research_recipe", "retireResearchRecipe", "RetireResearchRecipe"),
        ("create_research_run", "createResearchRun", "CreateResearchRun"),
        ("research_runs", "researchRuns", "ResearchRuns"),
        ("research_run_stats", "researchRunStats", "ResearchRunStats"),
        ("cancel_research_run", "cancelResearchRun", "CancelResearchRun"),
        ("research_run_events", "researchRunEvents", "ResearchRunEvents"),
        ("create_comparison_campaign", "createComparisonCampaign", "CreateComparisonCampaign"),
        ("comparison_campaign_events", "comparisonCampaignEvents", "ComparisonCampaignEvents"),
        ("research_results", "researchResults", "ResearchResults"),
        ("research_result_operator_audit", "researchResultOperatorAudit", "ResearchResultOperatorAudit"),
        ("research_result_decisions", "researchResultDecisions", "ResearchResultDecisions"),
        ("research_result_perspectives", "researchResultPerspectives", "ResearchResultPerspectives"),
        ("prepare_research_result_render", "prepareResearchResultRender", "PrepareResearchResultRender"),
        ("prepare_research_decision_render", "prepareResearchDecisionRender", "PrepareResearchDecisionRender"),
        ("laffat", "laffat", "Laffat"),
        ("laffa", "laffa", "Laffa"),
        ("prepare_laffa_render", "prepareLaffaRender", "PrepareLaffaRender"),
        ("laffa_renders", "laffaRenders", "LaffaRenders"),
    )
    for python_name, typescript_name, go_name in method_names:
        assert f"def {python_name}(" in python_source
        assert f"{typescript_name}(" in typescript_source
        assert f") {go_name}(" in go_source

    for route in (
        "/api/v2/domain-capabilities",
        "/api/v2/domain-compilations/",
        "/api/v2/rule-primitives",
        "/api/v2/research-recipes",
        "/api/v2/research-runs",
        "/api/v2/research-stats",
        "/api/v2/comparison-campaigns",
        "/api/v2/research-results",
        "/api/v2/laffat",
    ):
        assert route in python_source
        assert route in typescript_source
        assert route in go_source

    assert "class ResearchRenderPreparePayload(TypedDict):" in python_source
    assert "observerSeat: NotRequired[Literal[" in python_source
    assert "interface ResearchRenderPrepareRequest" in typescript_source
    assert "observerSeat?: 'north'|'west'|'south'|'east'|null" in typescript_source
    assert "type ResearchRenderPrepareRequest struct" in go_source
    assert "ObserverSeat ResearchObserverSeat" in go_source


def test_all_sdks_expose_the_recipe_mutation_and_page_contract() -> None:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(
        encoding="utf-8"
    )
    go_source = (ROOT / "sdk/go/client.go").read_text(encoding="utf-8")

    assert "class ResearchRecipeListResponse(TypedDict):" in python_source
    assert "items: list[dict[str, Any]]" in python_source
    assert "nextCursor: str | None" in python_source
    assert "idempotency_key: str | None = None" in python_source
    assert "cursor: str | None = None" in python_source
    assert "str(uuid4()) if value is None else value" in python_source

    assert "export interface ResearchRecipeListOptions" in typescript_source
    assert "export interface ResearchRecipeListResponse" in typescript_source
    assert "items: ObjectJson[]" in typescript_source
    assert "nextCursor: string|null" in typescript_source
    assert "idempotencyKey?:string" in typescript_source
    assert "value??crypto.randomUUID()" in typescript_source
    assert "options.cursor!==undefined" in typescript_source

    assert "type ResearchRecipeListResponse struct" in go_source
    assert "Items      []map[string]any" in go_source
    assert "NextCursor *string" in go_source
    assert "idempotencyKey ...string" in go_source
    assert "cursor ...string" in go_source
    assert '"crypto/rand"' in go_source
    assert '"encoding/hex"' in go_source
    assert "if len(values) > 1" in go_source


def test_all_sdks_submit_legacy_video_through_the_idempotent_v4_adapter() -> None:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(
        encoding="utf-8"
    )
    go_source = (ROOT / "sdk/go/client.go").read_text(encoding="utf-8")

    assert "idempotency_key: str | None = None" in python_source
    assert '"Idempotency-Key": _research_idempotency_key' in python_source
    assert "createVideo(payload:ObjectJson,idempotencyKey?:string)" in (
        typescript_source
    )
    assert "'Idempotency-Key':this.researchIdempotencyKey(idempotencyKey)" in (
        typescript_source
    )
    assert (
        "CreateVideo(ctx context.Context, payload, out any, idempotencyKey ...string)"
        in go_source
    )
    assert '"/api/v1/video-jobs", payload, out, map[string]string{' in go_source


def test_all_sdks_expose_optional_exact_bundle_selectors() -> None:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(
        encoding="utf-8"
    )
    go_source = (ROOT / "sdk/go/client.go").read_text(encoding="utf-8")

    assert python_source.count("bundle_revision_id: str | None = None") >= 2
    assert 'params["bundleRevisionId"] = bundle_revision_id' in python_source
    assert (
        "domainCapabilities(includePayloads=false,bundleRevisionId?:string)"
        in typescript_source
    )
    assert "rulePrimitives(bundleRevisionId?:string)" in typescript_source
    assert typescript_source.count("bundleRevisionId") >= 4
    assert go_source.count("bundleRevisionID ...string") == 2
    assert 'query.Set("bundleRevisionId", bundleRevisionID[0])' in go_source
    assert 'url.Values{"bundleRevisionId": []string{bundleRevisionID[0]}}' in go_source


def test_python_and_typescript_sdks_cover_strict_result_semantic_filters() -> None:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(
        encoding="utf-8"
    )
    filters = {
        "scenario_template": "scenarioTemplate",
        "scenario_origin": "scenarioOrigin",
        "scenario_status": "scenarioStatus",
        "requested_proof_mode": "requestedProofMode",
        "achieved_proof_mode": "achievedProofMode",
        "proof_completion": "proofCompletion",
        "proof_visibility": "proofVisibility",
        "target_team": "targetTeam",
        "contract_outcome": "contractOutcome",
        "project_kind": "projectKind",
        "project_value": "projectValue",
        "project_state": "projectState",
        "winning_root_action_band": "winningRootActionBand",
        "decision_kind": "decisionKind",
        "research_backend": "researchBackend",
        "qualification_authority": "qualificationAuthority",
        "runtime_compatibility_status": "runtimeCompatibilityStatus",
        "configuration_bundle": "configurationBundle",
        "exact_state_id": "exactStateId",
        "scenario_category": "scenarioCategory",
        "variant_collection": "variantCollection",
    }
    for python_name, api_name in filters.items():
        assert f"{python_name}: str | None = None" in python_source
        assert f'"{api_name}": {python_name}' in python_source
        assert f"{api_name}?:string" in typescript_source
    assert "exact_state_available: bool | None = None" in python_source
    assert '"exactStateAvailable": exact_state_available' in python_source
    assert "exactStateAvailable?:boolean" in typescript_source
