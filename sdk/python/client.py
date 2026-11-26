from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any, Literal, NotRequired, TypedDict, Unpack, cast
from uuid import uuid4

import httpx


class ResearchRenderPreparePayload(TypedDict):
    revealPolicy: NotRequired[
        Literal["after_commit", "after_completion", "post_reveal_only"]
    ]
    exportPreset: NotRequired[Literal["decision_clip", "hooked_flagship"] | None]
    observerSeat: NotRequired[Literal["north", "west", "south", "east"] | None]


class ResearchRecipeListResponse(TypedDict):
    items: list[dict[str, Any]]
    nextCursor: str | None


class IdentityItem(TypedDict):
    subjectId: str
    kind: str
    displayName: str
    username: str | None
    authenticated: bool
    guest: bool
    createdAt: str | None


class PendingRequestItem(TypedDict):
    requestId: str
    senderId: str
    recipientId: str
    status: Literal["pending"]
    createdAt: str
    senderDisplayName: str
    senderUsername: str | None


class IdentityPage(TypedDict):
    items: list[IdentityItem]
    nextCursor: str | None


class PendingRequestPage(TypedDict):
    items: list[PendingRequestItem]
    nextCursor: str | None


class ResearchAuthorityV1(TypedDict):
    schemaVersion: Literal[1]
    backendId: Literal["ortools_cp_sat_v2"]
    backendVersion: str
    qualificationAuthority: Literal["cp_sat"]
    solverVersion: str
    bundleRevisionId: str
    bundleHash: str
    domainIrHashes: dict[str, str]
    loweringManifestHash: str
    constructionModelHash: str
    proofModelHash: str
    graphHash: str
    traceHash: str
    proofOutcomeHash: str
    sha256: str


class RuntimeCompatibilityV1(TypedDict):
    schemaVersion: Literal[1]
    status: Literal["not_checked", "passed", "failed"]
    engineVersion: str | None
    bundleHash: str
    traceHash: str
    evidenceHash: str | None
    failureCodes: list[str]
    sha256: str


class ExecutionReceiptV1(TypedDict):
    schemaVersion: Literal[1]
    semanticResultHash: str
    stageTimings: dict[str, float | int]
    cpuSeconds: float | int
    peakMemoryBytes: int
    conflicts: int
    branches: int
    cacheBytes: int
    completedVariants: int
    machine: dict[str, Any]
    sha256: str


class ResearchOperatorAuditV1Response(TypedDict):
    schemaVersion: Literal[1]
    resultCode: str
    researchAuthority: ResearchAuthorityV1
    runtimeCompatibility: RuntimeCompatibilityV1
    executionReceipt: ExecutionReceiptV1


def _research_idempotency_key(value: str | None) -> str:
    return str(uuid4()) if value is None else value


class BalootClient:
    """Small synchronous client for games, analysis, Research, and rendering."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        api_key: str | None = None,
        trusted_key: str | None = None,
        subject_id: str | None = None,
        subject_name: str | None = None,
        timeout: float = 30.0,
    ):
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        if trusted_key:
            headers["x-baloot-trusted-key"] = trusted_key
        if subject_id:
            headers["x-baloot-subject-id"] = subject_id
        if subject_name:
            headers["x-baloot-subject-name"] = subject_name
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def __enter__(self) -> "BalootClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def bundle(self, mode: str = "full") -> dict[str, Any]:
        return self._get("/api/v1/bundles/current", params={"mode": mode})

    def health(self) -> dict[str, Any]:
        return self._get("/api/v1/health")

    def plugins(self) -> dict[str, Any]:
        return self._get("/api/v1/plugins")

    def list_plugins(self) -> dict[str, Any]:
        return self.plugins()

    def me(self) -> dict[str, Any]:
        return self._get("/api/v1/auth/me")

    def guest(self, display_name: str | None = None) -> dict[str, Any]:
        return self._post("/api/v1/auth/guest", {"displayName": display_name})

    def register(self, username: str, password: str, *, display_name: str | None = None) -> dict[str, Any]:
        return self._post("/api/v1/auth/register", {
            "username": username, "password": password, "displayName": display_name,
        })

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._post("/api/v1/auth/login", {"username": username, "password": password})

    def logout(self) -> None:
        response = self.client.post("/api/v1/auth/logout")
        response.raise_for_status()

    def friends(self) -> dict[str, Any]:
        return self._get("/api/v1/friends")

    def friends_page(self, *, cursor: str | None = None, limit: int | None = None) -> IdentityPage:
        return cast(IdentityPage, self._identity_page("/api/v2/friends", cursor=cursor, limit=limit))

    def friend_requests(self, *, cursor: str | None = None, limit: int | None = None) -> PendingRequestPage:
        return cast(PendingRequestPage, self._identity_page("/api/v2/friends/requests", cursor=cursor, limit=limit))

    def search_users(self, query: str, *, cursor: str | None = None, limit: int | None = None) -> IdentityPage:
        return cast(IdentityPage, self._identity_page("/api/v2/users:search", query=query, cursor=cursor, limit=limit))

    def _identity_page(self, path: str, **options: str | int | None) -> dict[str, Any]:
        params = {key: value for key, value in options.items() if value is not None}
        return self._get(path, params=params)

    def request_friend(self, recipient_id: str) -> dict[str, Any]:
        return self._post("/api/v1/friends/requests", {"recipientId": recipient_id})

    def validate_configuration(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/configuration:validate", {"request": request})

    def create_game(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/games", payload)

    def game(self, game_id: str, *, reveal_hands: bool = False) -> dict[str, Any]:
        return self._get(f"/api/v1/games/{game_id}", params={"revealHands": str(reveal_hands).lower()})

    def get_game(self, game_id: str, *, reveal_hands: bool = False) -> dict[str, Any]:
        return self.game(game_id, reveal_hands=reveal_hands)

    def bid(self, game_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/games/{game_id}/bids", payload)

    def play(self, game_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/games/{game_id}/actions", payload)

    def play_card(self, game_id: str, **payload: Any) -> dict[str, Any]:
        return self.play(game_id, **payload)

    def create_live_room(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/live/rooms", payload)

    def websocket_ticket(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/security/websocket-tickets", payload)

    def advance_bots(self, game_id: str, *, stop_before: str | None = "south", maximum_actions: int = 64) -> dict[str, Any]:
        params = {"maximumActions": maximum_actions}
        if stop_before is not None:
            params["stopBefore"] = stop_before
        return self._post(f"/api/v1/games/{game_id}/bots:advance", None, params=params)

    def next_round(self, game_id: str, *, seed: int | None = None) -> dict[str, Any]:
        return self._post(f"/api/v1/games/{game_id}/rounds:next", None, params={"seed": seed} if seed is not None else None)

    def audit(self, game_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/games/{game_id}/audit")

    def validate_context(self, state: dict[str, Any], *, rules_profile: str = "full") -> dict[str, Any]:
        return self._post("/api/v1/contexts:validate", {"state": state, "rulesProfile": rules_profile})

    def continue_context(self, state: dict[str, Any], *, rules_profile: str = "full") -> dict[str, Any]:
        return self._post("/api/v1/contexts:continue", {"state": state, "rulesProfile": rules_profile})

    def analyze(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/analyses", payload)

    def create_video(
        self,
        *,
        idempotency_key: str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Start the deprecated Research V4 compatibility adapter."""

        return self._post(
            "/api/v1/video-jobs",
            payload,
            headers={
                "Idempotency-Key": _research_idempotency_key(idempotency_key)
            },
        )

    def video_job(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/video-jobs/{job_id}")

    def cancel_video(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/video-jobs/{job_id}:cancel", None)

    def create_foundry_analysis_batch(self, campaign_hash: str) -> dict[str, Any]:
        return self._post(
            "/api/v2/analysis-batches", {"campaignHash": campaign_hash}
        )

    def foundry_analysis_batch(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/api/v2/analysis-batches/{job_id}")

    def cancel_foundry_analysis_batch(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/api/v2/analysis-batches/{job_id}:cancel", None)

    def create_render_batch(
        self,
        items: list[dict[str, Any]],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/render-batches",
            {"items": items},
            headers={"Idempotency-Key": idempotency_key},
        )

    def render_batch(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/api/v2/render-batches/{job_id}")

    def cancel_render_batch(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/api/v2/render-batches/{job_id}:cancel", None)

    def create_context_search(
        self,
        parent_trace_reference: dict[str, Any],
        decision_id: str,
        context_proof_hashes: list[str],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/context-searches",
            {
                "parentTraceReference": parent_trace_reference,
                "decisionId": decision_id,
                "contextProofHashes": context_proof_hashes,
            },
            headers={"Idempotency-Key": idempotency_key},
        )

    def context_search(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/api/v2/context-searches/{job_id}")

    def cancel_context_search(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/api/v2/context-searches/{job_id}:cancel", None)

    def create_natural_sakka_generation(
        self,
        root_seed: int,
        *,
        initial_dealer: str = "east",
        idempotency_key: str,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/sakka-generations",
            {"rootSeed": root_seed, "initialDealer": initial_dealer},
            headers={"Idempotency-Key": idempotency_key},
        )

    def natural_sakka_generation(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/api/v2/sakka-generations/{job_id}")

    def cancel_natural_sakka_generation(self, job_id: str) -> dict[str, Any]:
        return self._post(f"/api/v2/sakka-generations/{job_id}:cancel", None)

    def foundry_sakkas(
        self, *, provenance: str | None = None, after: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if provenance:
            params["provenance"] = provenance
        if after:
            params["after"] = after
        return self._get("/api/v2/sakkas", params=params)

    def create_natural_sakka(
        self, trace_references: list[dict[str, Any]], *, target: int = 152
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/sakkas/natural",
            {"traceReferences": trace_references, "target": target},
        )

    def create_curated_sakka(
        self, selections: list[dict[str, Any]], *, target: int = 152
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/sakkas/curated", {"selections": selections, "target": target}
        )

    def foundry_sakka(self, sakka_hash: str) -> dict[str, Any]:
        return self._get(f"/api/v2/sakkas/{sakka_hash}")

    def create_campaign_equivalence_proof(
        self, executions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/campaign-equivalence-proofs", {"executions": executions}
        )

    def foundry_campaign_attempts(self, campaign_hash: str) -> dict[str, Any]:
        return self._get(f"/api/v2/campaigns/{campaign_hash}/attempts")

    def foundry_profile(self) -> dict[str, Any]:
        return self._get("/api/v2/foundry-profile")

    def foundry_campaigns(
        self, *, after: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        return self._get("/api/v2/campaigns", params=params)

    def create_trace_import_catalog(
        self, lane: str, entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/trace-import-catalogs", {"lane": lane, "entries": entries}
        )

    def create_score_reachability_catalog(
        self, trace_references: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/score-reachability-catalogs",
            {"traceReferences": trace_references},
        )

    def prove_score_reachability(
        self,
        catalog_hash: str,
        target_score: tuple[int, int],
        *,
        target_dealer: str,
        initial_dealer: str = "east",
    ) -> dict[str, Any]:
        return self._post(
            f"/api/v2/score-reachability-catalogs/{catalog_hash}:prove",
            {
                "targetScore": list(target_score),
                "targetDealer": target_dealer,
                "initialDealer": initial_dealer,
            },
        )

    def foundry_catalogs(
        self,
        *,
        kind: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if kind:
            params["kind"] = kind
        if after:
            params["after"] = after
        return self._get("/api/v2/foundry-catalogs", params=params)

    def foundry_catalog(self, catalog_hash: str) -> dict[str, Any]:
        return self._get(f"/api/v2/foundry-catalogs/{catalog_hash}")

    def create_challenge_locator(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v2/challenge-locators", payload)

    def challenge_locator(self, code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/challenge-locators/{code}")

    def commit_challenge_locator_attempt(
        self, code: str, **payload: Any
    ) -> dict[str, Any]:
        return self._post(f"/api/v2/challenge-locators/{code}/attempts", payload)

    def revoke_challenge_locator(self, code: str) -> dict[str, Any]:
        return self._post(f"/api/v2/challenge-locators/{code}:revoke", None)

    # Unified domain compiler and Research Foundry V2.
    def domain_capabilities(
        self,
        *,
        include_payloads: bool = False,
        bundle_revision_id: str | None = None,
    ) -> dict[str, Any]:
        params = {"includePayloads": str(include_payloads).lower()}
        if bundle_revision_id is not None:
            params["bundleRevisionId"] = bundle_revision_id
        return self._get(
            "/api/v2/domain-capabilities",
            params=params,
        )

    def domain_compilation(
        self, compilation_hash: str, *, include_payloads: bool = False
    ) -> dict[str, Any]:
        return self._get(
            f"/api/v2/domain-compilations/{compilation_hash}",
            params={"includePayloads": str(include_payloads).lower()},
        )

    def domain_compilation_coverage(self, compilation_hash: str) -> dict[str, Any]:
        return self._get(f"/api/v2/domain-compilations/{compilation_hash}/coverage")

    def rule_primitives(
        self, *, bundle_revision_id: str | None = None
    ) -> dict[str, Any]:
        params = {}
        if bundle_revision_id is not None:
            params["bundleRevisionId"] = bundle_revision_id
        return self._get("/api/v2/rule-primitives", params=params)

    def validate_research_recipe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v2/research-recipes:validate", payload)

    def estimate_research_recipe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v2/research-recipes:estimate", payload)

    def create_research_recipe(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/research-recipes",
            payload,
            headers={
                "Idempotency-Key": _research_idempotency_key(idempotency_key)
            },
        )

    def research_recipes(
        self,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ResearchRecipeListResponse:
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        if cursor is not None:
            params["cursor"] = cursor
        return self._get("/api/v2/research-recipes", params=params)

    def research_recipe(self, code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/research-recipes/{code}")

    def retire_research_recipe(
        self, code: str, *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return self._post(
            f"/api/v2/research-recipes/{code}:retire",
            None,
            headers={
                "Idempotency-Key": _research_idempotency_key(idempotency_key)
            },
        )

    def create_research_run(
        self, recipe_code: str, *, idempotency_key: str
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/research-runs",
            {"recipeCode": recipe_code},
            headers={"Idempotency-Key": idempotency_key},
        )

    def research_runs(
        self, *, status: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        return self._get("/api/v2/research-runs", params=params)

    def research_run(self, run_code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/research-runs/{run_code}")

    def research_run_stats(self, run_code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/research-runs/{run_code}/stats")

    def research_stats(self) -> dict[str, Any]:
        return self._get("/api/v2/research-stats")

    def cancel_research_run(self, run_code: str) -> dict[str, Any]:
        return self._post(f"/api/v2/research-runs/{run_code}:cancel", None)

    def research_run_events(self, run_code: str) -> Iterator[dict[str, Any]]:
        yield from self._events(f"/api/v2/research-runs/{run_code}/events")

    def create_comparison_campaign(
        self, recipe_code: str, *, idempotency_key: str
    ) -> dict[str, Any]:
        return self._post(
            "/api/v2/comparison-campaigns",
            {"recipeCode": recipe_code},
            headers={"Idempotency-Key": idempotency_key},
        )

    def comparison_campaigns(self, *, limit: int = 100) -> dict[str, Any]:
        return self._get("/api/v2/comparison-campaigns", params={"limit": limit})

    def comparison_campaign(self, campaign_code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/comparison-campaigns/{campaign_code}")

    def comparison_campaign_events(
        self, campaign_code: str
    ) -> Iterator[dict[str, Any]]:
        yield from self._events(
            f"/api/v2/comparison-campaigns/{campaign_code}/events"
        )

    def research_results(
        self,
        *,
        run_code: str | None = None,
        truth_label: str | None = None,
        disposition: str | None = None,
        minimum_score: float | None = None,
        search: str | None = None,
        tactical_tag: str | None = None,
        acting_seat: str | None = None,
        observer_seat: str | None = None,
        contract: str | None = None,
        project: str | None = None,
        generation_lane: str | None = None,
        scenario_template: str | None = None,
        scenario_origin: str | None = None,
        scenario_status: str | None = None,
        requested_proof_mode: str | None = None,
        achieved_proof_mode: str | None = None,
        proof_completion: str | None = None,
        proof_visibility: str | None = None,
        target_team: str | None = None,
        contract_outcome: str | None = None,
        project_kind: str | None = None,
        project_value: str | None = None,
        project_state: str | None = None,
        winning_root_action_band: str | None = None,
        exact_state_available: bool | None = None,
        decision_kind: str | None = None,
        research_backend: str | None = None,
        qualification_authority: str | None = None,
        runtime_compatibility_status: str | None = None,
        configuration_bundle: str | None = None,
        exact_state_id: str | None = None,
        scenario_category: str | None = None,
        variant_collection: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        group_by: str | None = None,
        sort: str = "score_desc",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        params = {
            "runCode": run_code,
            "truthLabel": truth_label,
            "disposition": disposition,
            "minimumScore": minimum_score,
            "search": search,
            "tacticalTag": tactical_tag,
            "actingSeat": acting_seat,
            "observerSeat": observer_seat,
            "contract": contract,
            "project": project,
            "generationLane": generation_lane,
            "scenarioTemplate": scenario_template,
            "scenarioOrigin": scenario_origin,
            "scenarioStatus": scenario_status,
            "requestedProofMode": requested_proof_mode,
            "achievedProofMode": achieved_proof_mode,
            "proofCompletion": proof_completion,
            "proofVisibility": proof_visibility,
            "targetTeam": target_team,
            "contractOutcome": contract_outcome,
            "projectKind": project_kind,
            "projectValue": project_value,
            "projectState": project_state,
            "winningRootActionBand": winning_root_action_band,
            "exactStateAvailable": exact_state_available,
            "decisionKind": decision_kind,
            "researchBackend": research_backend,
            "qualificationAuthority": qualification_authority,
            "runtimeCompatibilityStatus": runtime_compatibility_status,
            "configurationBundle": configuration_bundle,
            "exactStateId": exact_state_id,
            "scenarioCategory": scenario_category,
            "variantCollection": variant_collection,
            "createdFrom": created_from,
            "createdTo": created_to,
            "groupBy": group_by,
            "sort": sort,
            "offset": offset,
            "limit": limit,
        }
        params = {key: value for key, value in params.items() if value is not None}
        return self._get("/api/v2/research-results", params=params)

    def research_result(self, code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/research-results/{code}")

    def research_result_operator_audit(
        self, code: str
    ) -> ResearchOperatorAuditV1Response:
        return self._get(f"/api/v2/research-results/{code}/operator-audit")

    def research_result_decisions(self, code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/research-results/{code}/decisions")

    def research_result_perspectives(
        self, code: str, observer_seat: str
    ) -> dict[str, Any]:
        return self._get(
            f"/api/v2/research-results/{code}/perspectives",
            params={"observerSeat": observer_seat},
        )

    def research_result_decision(self, code: str, ordinal: int) -> dict[str, Any]:
        return self._get(f"/api/v2/research-results/{code}/decisions/{ordinal}")

    def prepare_research_result_render(
        self, code: str, **payload: Unpack[ResearchRenderPreparePayload]
    ) -> dict[str, Any]:
        return self._post(f"/api/v2/research-results/{code}:prepare-render", payload)

    def prepare_research_decision_render(
        self,
        code: str,
        ordinal: int,
        **payload: Unpack[ResearchRenderPreparePayload],
    ) -> dict[str, Any]:
        return self._post(
            f"/api/v2/research-results/{code}/decisions/{ordinal}:prepare-render",
            payload,
        )

    def research_result_renders(self, code: str) -> dict[str, Any]:
        return self._get(f"/api/v2/research-results/{code}/renders")

    def laffat(
        self,
        *,
        search: str | None = None,
        source_kind: str | None = None,
        replay_valid: bool | None = None,
        qualification_status: str | None = None,
        truth_label: str | None = None,
        contract: str | None = None,
        project: str | None = None,
        tactic: str | None = None,
        seat: str | None = None,
        render_status: str | None = None,
        sort: str = "newest",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        params = {
            "search": search,
            "sourceKind": source_kind,
            "replayValid": replay_valid,
            "qualificationStatus": qualification_status,
            "truthLabel": truth_label,
            "contract": contract,
            "project": project,
            "tactic": tactic,
            "seat": seat,
            "renderStatus": render_status,
            "sort": sort,
            "offset": offset,
            "limit": limit,
        }
        params = {key: value for key, value in params.items() if value is not None}
        return self._get("/api/v2/laffat", params=params)

    def laffa(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v2/laffat/{identifier}")

    def prepare_laffa_render(
        self, identifier: str, **payload: Unpack[ResearchRenderPreparePayload]
    ) -> dict[str, Any]:
        return self._post(f"/api/v2/laffat/{identifier}:prepare-render", payload)

    def laffa_renders(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v2/laffat/{identifier}/renders")

    def archived_game(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/history/games/{identifier}")

    def replay_package(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/history/games/{identifier}/replay")

    def audit_game(self, identifier: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/history/games/{identifier}:audit", payload)

    def explain_decision(self, decision_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/history/decisions/{decision_id}:explain", payload)

    def compare_decision(self, decision_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/history/decisions/{decision_id}/comparison")

    def annotate_intent(self, decision_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/history/decisions/{decision_id}/intent", payload)

    def record_decision_attempt(self, decision_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/history/decisions/{decision_id}/attempts", payload)

    def decision_difficulty(self, decision_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/history/decisions/{decision_id}/difficulty")

    def record_challenge_attempt(self, challenge_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/challenges/{challenge_id}/attempts", payload)

    def challenge_difficulty(self, challenge_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/challenges/{challenge_id}/difficulty")

    def adaptive_challenge(self, *, player_id: str | None = None, recent: list[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"recent": ",".join(recent or [])}
        if player_id:
            params["playerId"] = player_id
        return self._get("/api/v1/adaptive/challenges/next", params=params)

    def video_record(self, video_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/videos/{video_id}")

    def video_timeline(self, video_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/videos/{video_id}/timeline")

    def video_audit_map(self, video_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/videos/{video_id}/audit-map")

    # Canonical shared standardized environment: لفة (one round) or صكّة (full match).
    def shared_boards(self, *, limit: int = 100) -> dict[str, Any]:
        return self._get("/api/v1/shared/boards", params={"limit": limit})

    def create_shared_board(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/shared/boards", payload)

    def shared_board(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/boards/{identifier}")

    def verify_shared_board(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/boards/{identifier}/verification")

    def create_shared_room(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/shared/rooms", payload)

    def shared_room(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/rooms/{identifier}")

    def join_shared_room(self, identifier: str) -> dict[str, Any]:
        return self._post(f"/api/v1/shared/rooms/{identifier}:join", {})

    def start_shared_run(self, identifier: str) -> dict[str, Any]:
        return self._post(f"/api/v1/shared/rooms/{identifier}/runs", {})

    def shared_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/runs/{run_id}")

    def verify_shared_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/runs/{run_id}/verification")

    def play_shared_action(self, run_id: str, action: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/api/v1/shared/runs/{run_id}/actions", action)

    def shared_run_review(self, run_id: str, *, ensure_audit: bool = False) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/runs/{run_id}/review", params={"ensureAudit": str(ensure_audit).lower()})

    def explain_shared_decision(self, run_id: str, decision_id: str, *, level: str = "basic",
                                surface: str = "replay", challenge_revealed: bool = False,
                                ensure_audit: bool = False, view_mode: str = "observer") -> dict[str, Any]:
        return self._get(
            f"/api/v1/shared/runs/{run_id}/decisions/{decision_id}/explanation",
            params={"level": level, "surface": surface,
                    "challengeRevealed": str(challenge_revealed).lower(),
                    "ensureAudit": str(ensure_audit).lower(), "viewMode": view_mode},
        )

    def shared_decision_reviews(self, run_id: str, decision_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/runs/{run_id}/decisions/{decision_id}/reviews")

    def review_shared_decision(self, run_id: str, decision_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/shared/runs/{run_id}/decisions/{decision_id}/reviews", payload)

    def share_shared_decision(self, run_id: str, sequence: int, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/shared/runs/{run_id}/actions/{sequence}:share", payload)

    def shared_leaderboard(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/shared/rooms/{identifier}/leaderboard")

    def compare_shared_runs(self, run_a: str, run_b: str) -> dict[str, Any]:
        return self._post("/api/v1/shared/comparisons", {"runA": run_a, "runB": run_b})

    # Shareable same-start Sakka (صكّة).
    def sakka_boards(self, *, limit: int = 100) -> dict[str, Any]:
        return self._get("/api/v1/sakka/boards", params={"limit": limit})

    def create_sakka_board(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/sakka/boards", payload)

    def sakka_board(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/sakka/boards/{identifier}")

    def verify_sakka_board(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/sakka/boards/{identifier}/verification")

    def create_sakka_room(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/sakka/rooms", payload)

    def sakka_room(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/sakka/rooms/{identifier}")

    def join_sakka_room(self, identifier: str) -> dict[str, Any]:
        return self._post(f"/api/v1/sakka/rooms/{identifier}:join", {})

    def start_sakka_run(self, identifier: str) -> dict[str, Any]:
        return self._post(f"/api/v1/sakka/rooms/{identifier}/runs", {})

    def sakka_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/sakka/runs/{run_id}")

    def verify_sakka_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/sakka/runs/{run_id}/verification")

    def play_sakka_action(self, run_id: str, action: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/api/v1/sakka/runs/{run_id}/actions", action)

    def audit_sakka_run(self, run_id: str, *, deep: bool = False,
                        force: bool = False, maximum_decisions: int = 32) -> dict[str, Any]:
        return self._post(f"/api/v1/sakka/runs/{run_id}:audit", {
            "deep": deep, "force": force, "maximumDecisions": maximum_decisions,
        })

    def share_sakka_decision(self, run_id: str, sequence: int, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/sakka/runs/{run_id}/actions/{sequence}:share", payload)

    def sakka_feedback(self, run_id: str, sequence: int, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/sakka/runs/{run_id}/actions/{sequence}/feedback", payload)

    def sakka_leaderboard(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/api/v1/sakka/rooms/{identifier}/leaderboard")

    def compare_sakka_runs(self, run_a: str, run_b: str) -> dict[str, Any]:
        return self._post("/api/v1/sakka/comparisons", {"runA": run_a, "runB": run_b})

    # Governed Human-in-the-Loop V18.
    def hitl_profile(self) -> dict[str, Any]:
        return self._get("/api/v1/hitl/profile")

    def hitl_cases(self, *, limit: int = 100) -> dict[str, Any]:
        return self._get("/api/v1/hitl/cases", params={"limit": limit})

    def create_hitl_case(self, decision_id: str, *, correction_type: str = "analysis_defect", high_impact: bool = False) -> dict[str, Any]:
        return self._post("/api/v1/hitl/cases", {"decisionId": decision_id, "correctionType": correction_type, "highImpact": high_impact})

    def hitl_case(self, case_id: str, *, blind: bool = False) -> dict[str, Any]:
        return self._get(f"/api/v1/hitl/cases/{case_id}", params={"blind": str(blind).lower()})

    def assign_hitl_review(self, case_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/cases/{case_id}/assignments", payload)

    def submit_hitl_review(self, case_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/cases/{case_id}/submissions", payload)

    def compute_hitl_consensus(self, case_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/cases/{case_id}/consensus", {})

    def hitl_consent(self) -> dict[str, Any]:
        return self._get("/api/v1/hitl/consent")

    def set_hitl_consent(self, **payload: Any) -> dict[str, Any]:
        response = self.client.put("/api/v1/hitl/consent", json=payload); response.raise_for_status(); return response.json()

    def create_hitl_training_candidate(self, case_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/cases/{case_id}/training-candidates", {})

    def evaluate_hitl_candidate(self, candidate_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/training-candidates/{candidate_id}/evaluations", payload)

    def render_hitl_review_clip(self, case_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/cases/{case_id}/review-clip", payload)

    def hitl_improvements(self) -> dict[str, Any]:
        return self._get("/api/v1/hitl/improvements")

    def create_hitl_improvement(self, **payload: Any) -> dict[str, Any]:
        return self._post("/api/v1/hitl/improvements", payload)

    def evaluate_hitl_improvement(self, proposal_id: str, **payload: Any) -> dict[str, Any]:
        return self._post(f"/api/v1/hitl/improvements/{proposal_id}/evaluations", payload)

    # Versioned row-based configuration control plane V19.
    def configuration_capabilities(self) -> dict[str, Any]:
        return self._get("/api/v1/configuration-registry/profile")

    def configuration_profiles(self, *, family: str | None = None, profile_id: str | None = None, status: str | None = None, tenant_id: str | None = None, scope: str | None = None, limit: int = 200) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if family is not None: params["family"] = family
        if profile_id is not None: params["profileId"] = profile_id
        if status is not None: params["status"] = status
        if tenant_id is not None: params["tenantId"] = tenant_id
        if scope is not None: params["scope"] = scope
        return self._get("/api/v1/configuration-registry/profiles", params=params)

    def create_configuration_profile(self, family: str, payload: dict[str, Any], *, scope: str = "global", tenant_id: str | None = None, supersedes_revision_id: str | None = None, publish: bool = False) -> dict[str, Any]:
        return self._post("/api/v1/configuration-registry/profiles", {"family": family, "payload": payload, "scope": scope, "tenantId": tenant_id, "supersedesRevisionId": supersedes_revision_id, "publish": publish})

    def publish_configuration_profile(self, revision_id: str, expected_etag: int) -> dict[str, Any]:
        return self._post(f"/api/v1/configuration-registry/profiles/{revision_id}:publish", {"expectedEtag": expected_etag})

    def reject_configuration_profile(self, revision_id: str, expected_etag: int, reason: str) -> dict[str, Any]:
        return self._post(f"/api/v1/configuration-registry/profiles/{revision_id}:reject", {"expectedEtag": expected_etag, "reason": reason})

    def retire_configuration_profile(self, revision_id: str, expected_etag: int, reason: str) -> dict[str, Any]:
        return self._post(f"/api/v1/configuration-registry/profiles/{revision_id}:retire", {"expectedEtag": expected_etag, "reason": reason})

    def resolve_configuration_profile(self, family: str, profile_id: str, *, tenant_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"family": family, "profileId": profile_id}
        if tenant_id is not None: params["tenantId"] = tenant_id
        return self._get("/api/v1/configuration-registry/profiles:resolve", params=params)

    def compare_configuration_profiles(self, left: str, right: str) -> dict[str, Any]:
        return self._get("/api/v1/configuration-registry/profiles:compare", params={"left": left, "right": right})

    def configuration_bundles(self, *, bundle_id: str | None = None, status: str | None = None, tenant_id: str | None = None, scope: str | None = None, limit: int = 200) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if bundle_id is not None: params["bundleId"] = bundle_id
        if status is not None: params["status"] = status
        if tenant_id is not None: params["tenantId"] = tenant_id
        if scope is not None: params["scope"] = scope
        return self._get("/api/v1/configuration-registry/bundles", params=params)

    def create_configuration_bundle(self, bundle_id: str, version: str, component_revisions: dict[str, str], *, mode: str = "full", scope: str = "global", tenant_id: str | None = None, supersedes_revision_id: str | None = None, publish: bool = False, allow_draft_components: bool = False) -> dict[str, Any]:
        return self._post("/api/v1/configuration-registry/bundles", {"bundleId": bundle_id, "version": version, "mode": mode, "componentRevisions": component_revisions, "scope": scope, "tenantId": tenant_id, "supersedesRevisionId": supersedes_revision_id, "publish": publish, "allowDraftComponents": allow_draft_components})

    def publish_configuration_bundle(self, revision_id: str, expected_etag: int) -> dict[str, Any]:
        return self._post(f"/api/v1/configuration-registry/bundles/{revision_id}:publish", {"expectedEtag": expected_etag})

    def publish_configuration(self, revision_id: str, expected_etag: int) -> dict[str, Any]:
        return self.publish_configuration_bundle(revision_id, expected_etag)

    def reject_configuration_bundle(self, revision_id: str, expected_etag: int, reason: str) -> dict[str, Any]:
        return self._post(f"/api/v1/configuration-registry/bundles/{revision_id}:reject", {"expectedEtag": expected_etag, "reason": reason})

    def retire_configuration_bundle(self, revision_id: str, expected_etag: int, reason: str) -> dict[str, Any]:
        return self._post(f"/api/v1/configuration-registry/bundles/{revision_id}:retire", {"expectedEtag": expected_etag, "reason": reason})

    def resolve_configuration_bundle(self, *, bundle_id: str | None = None, revision_id: str | None = None, tenant_id: str | None = None, allow_draft: bool = False) -> dict[str, Any]:
        if revision_id is not None:
            return self._get(f"/api/v1/configuration-registry/bundles/{revision_id}:resolve", params={"allowDraft": str(allow_draft).lower()})
        params: dict[str, Any] = {}
        if bundle_id is not None: params["bundleId"] = bundle_id
        if tenant_id is not None: params["tenantId"] = tenant_id
        return self._get("/api/v1/configuration-registry/resolve", params=params)

    def resolve_configuration(self, *, bundle_id: str | None = None, revision_id: str | None = None, tenant_id: str | None = None, allow_draft: bool = False) -> dict[str, Any]:
        return self.resolve_configuration_bundle(
            bundle_id=bundle_id,
            revision_id=revision_id,
            tenant_id=tenant_id,
            allow_draft=allow_draft,
        )

    def plugins(self, *, kind: str | None = None) -> dict[str, Any]:
        return self._get("/api/v1/plugins", params={"kind": kind} if kind else None)

    def platform_capabilities(self) -> dict[str, Any]:
        return self._get("/api/v1/platform/capabilities")

    def performance_profile(self) -> dict[str, Any]:
        return self._get("/api/v1/performance/profile")

    def run_performance_quick(self) -> dict[str, Any]:
        return self._post("/api/v1/performance/quick", {})

    def wait_video(self, job_id: str, *, interval: float = .5, timeout: float = 300) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.video_job(job_id)
            if job["status"] in {"completed", "failed", "cancelled"}:
                return job
            time.sleep(interval)
        raise TimeoutError(job_id)

    def _get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.client.get(path, **kwargs)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
        response = self.client.post(path, json=payload, **kwargs)
        response.raise_for_status()
        return response.json()

    def _events(self, path: str) -> Iterator[dict[str, Any]]:
        with self.client.stream("GET", path) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])
