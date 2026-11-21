from __future__ import annotations

import copy
import html
import hashlib
import math
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    RootModel,
    StrictBool,
    field_validator,
    model_validator,
    model_serializer,
)
from pydantic.json_schema import SkipJsonSchema

from baloot_platform.config import build_bundle, profile_bundle_from_resolved
from baloot_platform.guide import build_engine_guide
from baloot_platform.analysis.belief_v7 import (
    BeliefPolicy,
    hidden_deal_attempt_budget,
)
from baloot_platform.analysis.search_errors import SearchTimeBudgetExceeded
from baloot_platform.version import __version__
from baloot_platform.research.family_catalog import (
    TAG_AXES,
    TagQueryV1,
    build_family_generation_request_v1,
    tag_query_schema_v1,
)
from baloot_platform.adjudication.research import AuthoredAdjudicationEventV1
from baloot_platform.observability import TELEMETRY
from baloot_platform.challenge.miner import MiningRequest
from baloot_platform.challenge.campaign import (
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
    SimulationShardSpecV1,
)
from baloot_platform.challenge.deep_analysis import FoundryDeepAnalysisWorkerV1
from baloot_platform.challenge.video_production_batch import FoundryVideoBatchServiceV1
from baloot_platform.challenge.worker import (
    DisagreementFoundryWorkerV1,
    GuidedFoundryWorkerV1,
    OrganicFoundryWorkerV1,
)
from baloot_platform.challenge.locator import ChallengeAttemptConflictError
from baloot_platform.audit import build_video_audit_map
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.core.match_scenario import (
    MatchScenarioRequest,
    generate_match_episode,
)
from baloot_platform.core.serialization import (
    continue_trusted_context,
    replay_trusted_context,
)
from baloot_platform.plugins.registry import PluginRegistry
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
)
from baloot_platform.service.routes import (
    build_adjudication_router,
    build_auth_router,
    build_canonical_history_router,
    build_live_router,
    build_sakka_router,
    build_security_router,
    build_shared_router,
    build_hitl_router,
    build_configuration_router,
    build_platform_v19_router,
    build_foundry_admin_v2_router,
)
from baloot_platform.service.container import PlatformContainer, build_container
from baloot_platform.identity.ownership import SERVICE_SUBJECT_OWNER
from baloot_platform.service.client_identity import (
    initialize_service_subject_ownership,
    service_subject_ownership,
)
from baloot_platform.service.routes.auth import request_token, require_identity
from baloot_platform.shared import (
    SharedDecisionAccessError,
    project_shared_decision,
    resolve_shared_decision_access,
    resolve_shared_game_access,
)
from baloot_platform.learning.challenge_publication import validate_reviewer_role
from baloot_platform.service.game import GameManager
from baloot_platform.service.laffa_library import LaffaLibraryServiceV1
from baloot_platform.video.card_registry import CardPackRegistry
from baloot_platform.research.contracts import (
    build_research_configuration_selector_v1,
    RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
    RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
)
from baloot_platform.research.recipe_api import (
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENCY_KEY_PATTERN,
    RECIPE_LIST_CURSOR_MAX_LENGTH,
    ResearchMutationConflictError,
    validate_idempotency_key,
)
from baloot_platform.research.service import (
    ResearchLaunchAdmissionError,
    _load_owned_result_json_artifact_v1,
    _operator_research_run_v1,
)
from baloot_platform.research.scenario_contracts import (
    PLAYING_QUALITY_MODES, PROOF_UTILITIES, SCENARIO_DECISION_KINDS,
    ScenarioPlayingQualityPolicyV1,
    ScenarioOriginV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_result import public_scenario_result_v4
from baloot_platform.research.result_authority import (
    ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES,
    ExecutionReceiptV1 as ExecutionReceiptContractV1,
    ResearchAuthorityV1 as ResearchAuthorityContractV1,
    RuntimeCompatibilityV1 as RuntimeCompatibilityContractV1,
)

CONTAINER: PlatformContainer = build_container()
ROOT = CONTAINER.root
DATA_ROOT = CONTAINER.data_root
OUTPUTS = CONTAINER.outputs
RUNTIME = CONTAINER.runtime
DATABASE_URL = CONTAINER.database_url
STORE = CONTAINER.jobs
EXECUTOR = CONTAINER.executor
REGISTRY = CONTAINER.registry
GAME_STORE = CONTAINER.game_store
LEARNING_STORE = CONTAINER.learning_store
AGENT_REGISTRY = CONTAINER.agent_registry
GAMES = CONTAINER.games
LEARNING = CONTAINER.learning
IDENTITY_PROFILE = CONTAINER.identity_profile
IDENTITY = CONTAINER.identity
_DUPLICATE_PROFILE = CONTAINER.duplicate_profile
SAKKA = CONTAINER.sakka
AUDIT = CONTAINER.audit
DIFFICULTY = CONTAINER.difficulty
CHALLENGE_MINER = CONTAINER.challenge_miner
FOUNDRY_CAMPAIGNS = CONTAINER.campaign_control
CHALLENGE_LOCATORS = CONTAINER.challenge_locators
CHALLENGE_PUBLICATION = CONTAINER.challenge_publication
FOUNDRY_CATALOGS = CONTAINER.foundry_catalogs
FOUNDRY_DEEP_ANALYSIS = CONTAINER.deep_analysis
FOUNDRY_ANALYSIS_CANDIDATES = CONTAINER.analysis_candidates
FOUNDRY_FINALIST_VALIDATIONS = CONTAINER.finalist_validations
FOUNDRY_CAMPAIGN_PROOFS = CONTAINER.campaign_proofs
FOUNDRY_PROFILE = CONTAINER.foundry_profile
RENDER_BATCHES = CONTAINER.render_batches
CONTEXT_SEARCHES = CONTAINER.context_searches
SAKKA_GENERATIONS = CONTAINER.sakka_generations
RESEARCH = CONTAINER.research
RESEARCH_RENDERS = CONTAINER.research_renders
_LOCAL_FOUNDRY_LOCK = threading.Lock()
_LOCAL_FOUNDRY_CAMPAIGNS: set[str] = set()
_LOCAL_ANALYSIS_WORKER_ACTIVE = False
ACTIVE_BUNDLE = profile_bundle_from_resolved(
    CONTAINER.configuration.resolve(
        revision_id=FOUNDRY_PROFILE["runtimeBundleRevisionId"]
    ),
    root=ROOT,
)
LAFFA_LIBRARY = LaffaLibraryServiceV1(
    learning_store=LEARNING_STORE,
    research=RESEARCH,
    foundry_catalogs=FOUNDRY_CATALOGS,
    render_batches=RENDER_BATCHES,
    active_bundle_hash=ACTIVE_BUNDLE.sha256,
)
ENGINE_GUIDE = build_engine_guide(ACTIVE_BUNDLE)
_VIDEO_LOCALIZATIONS = dict(ACTIVE_BUNDLE.video["localizations"])
_VIDEO_JOB_TITLE_DEFAULT = str(_VIDEO_LOCALIZATIONS["video.decision.challenge_title"])
_VIDEO_JOB_QUESTION_DEFAULT = str(
    _VIDEO_LOCALIZATIONS["video.decision.challenge_question"]
)


def _shared_decision_access(
    decision: Mapping[str, Any],
    request: Request,
) -> tuple[Any, str | None]:
    try:
        access = resolve_shared_decision_access(SAKKA.store, decision)
    except SharedDecisionAccessError as exc:
        raise HTTPException(403, str(exc)) from exc
    if access is None or access.completed:
        return access, None
    try:
        identity, _session = require_identity(request, IDENTITY)
    except HTTPException:
        return access, None
    return access, identity.subject_id


def _project_decision_for_request(
    decision: Mapping[str, Any],
    request: Request,
) -> dict[str, Any]:
    access, subject_id = _shared_decision_access(decision, request)
    return project_shared_decision(decision, access, subject_id=subject_id)


def _require_private_decision_access(
    decision_id: str,
    request: Request,
) -> dict[str, Any]:
    decision = LEARNING_STORE.get_decision(decision_id)
    if decision is None:
        raise HTTPException(404, "decision not found")
    access, subject_id = _shared_decision_access(decision, request)
    if access is not None and not access.can_view_private(subject_id):
        raise HTTPException(403, "private decision analysis is unavailable during an active shared run")
    return decision


def _require_completed_shared_game_access(game_id: str) -> None:
    try:
        access = resolve_shared_game_access(SAKKA.store, game_id)
    except SharedDecisionAccessError as exc:
        raise HTTPException(403, str(exc)) from exc
    if access is not None and not access.completed:
        raise HTTPException(403, "aggregate game analysis is unavailable during an active shared run")


def _require_private_history_game(identifier: str) -> dict[str, Any]:
    game = LEARNING_STORE.get_game(identifier)
    if game is None:
        raise HTTPException(404, "archived game not found")
    _require_completed_shared_game_access(str(game.get("gameId") or ""))
    _require_completed_shared_analysis_decisions(game)
    return game


def _require_completed_shared_analysis_decisions(value: Mapping[str, Any]) -> None:
    for item in value.get("decisions") or []:
        decision_id = str(item.get("decisionId") or "")
        decision = LEARNING_STORE.get_decision(decision_id)
        if decision is None:
            raise HTTPException(403, "aggregate decision identity is unavailable")
        try:
            access = resolve_shared_decision_access(SAKKA.store, decision)
        except SharedDecisionAccessError as exc:
            raise HTTPException(403, str(exc)) from exc
        if access is not None and not access.completed:
            raise HTTPException(403, "aggregate game analysis is unavailable during an active shared run")
BELIEF_POLICY = BeliefPolicy.from_profile(
    ACTIVE_BUNDLE.component("belief"),
    root=ROOT,
)
_FOUNDRY_JOB_POLICY = dict(ACTIVE_BUNDLE.component("database")["jobs"])
_CHALLENGE_REQUEST_DEFAULTS = dict(CHALLENGE_MINER.profile["request_defaults"])
_CHALLENGE_CONTENT_CONTRACTS = {
    "auto",
    "any",
    *CHALLENGE_MINER.contract_profile.raw["contracts"],
}
_CHALLENGE_PUBLICATION_POLICY = dict(CHALLENGE_MINER.profile["publication"])
EXPLANATIONS = CONTAINER.explanations
REVIEWS = CONTAINER.reviews
CANONICAL_HISTORY = CONTAINER.canonical_history
ADJUDICATION = CONTAINER.adjudication
LIVE = CONTAINER.live
AUTHORIZATION_POLICY = CONTAINER.authorization_policy
ACCESS_TOKENS = CONTAINER.access_tokens
SECURITY_STATE = CONTAINER.security_state
OIDC_TRANSACTIONS = CONTAINER.oidc_transactions
OIDC_PROVIDERS = CONTAINER.oidc_providers
OIDC_CALLBACK = CONTAINER.oidc_callback
PASSKEYS = CONTAINER.passkeys
WEBSOCKET_TICKETS = CONTAINER.websocket_tickets
HITL = CONTAINER.hitl
HITL_REVIEW_RENDERER = CONTAINER.hitl_review_renderer
HITL_PROFILE = CONTAINER.hitl_profile
CERTIFICATION_PROFILE = CONTAINER.certification_profile
BENCHMARK_PROFILE = CONTAINER.benchmark_profile
OPERATIONS_PROFILE = CONTAINER.operations_profile
CONFIGURATION = CONTAINER.configuration
CONFIGURATION_PROFILE = CONTAINER.configuration_profile
PERFORMANCE_PROFILE = CONTAINER.performance_profile
PERFORMANCE = CONTAINER.performance
CAPABILITY_MATRIX = CONTAINER.capability_matrix
CARD_PACKS = CardPackRegistry.from_yaml("configs/assets/card_packs.yaml", root=ROOT)

_RATE_LOCK = threading.RLock()
_RATE_EVENTS: dict[str, deque[float]] = defaultdict(deque)
_METRIC_LOCK = threading.RLock()
_METRICS: dict[tuple[str, str, int], int] = defaultdict(int)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    initialize_service_subject_ownership()
    # In-process recovery is deliberately local-only. Distributed deployments
    # use separately supervised, leased workers so multiple API replicas cannot
    # execute the same persisted job concurrently.
    if not DATABASE_URL:
        _recover_local_foundry_work()
    yield
    # Test clients and short-lived workers must not leak executor threads.
    if os.environ.get("BALOOT_SHUTDOWN_CONTAINER_ON_APP_EXIT", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        CONTAINER.shutdown()


app = FastAPI(
    title="Baloot Platform",
    version=__version__,
    description="Authoritative configurable Saudi Baloot lifecycle, analysis, shared لفة/صكّة, learning and video platform",
    lifespan=_lifespan,
    docs_url=None,
    redoc_url=None,
)


@app.exception_handler(SearchTimeBudgetExceeded)
async def search_time_budget_exceeded(_request: Request, exc: SearchTimeBudgetExceeded):
    return JSONResponse(status_code=409, content={"detail": exc.to_detail()})


app.state.identity_service = IDENTITY
app.state.sakka_service = SAKKA
app.state.shared_service = SAKKA
app.state.explanation_service = EXPLANATIONS
app.state.review_service = REVIEWS
app.state.canonical_history = CANONICAL_HISTORY


@app.get("/docs", include_in_schema=False)
@app.get("/redoc", include_in_schema=False)
def api_documentation():
    return FileResponse(Path(__file__).resolve().parent / "static" / "api_docs.html")


app.state.adjudication_service = ADJUDICATION
app.state.live_service = LIVE
app.state.hitl_service = HITL
app.state.authorization_policy = AUTHORIZATION_POLICY
app.state.configuration_service = CONFIGURATION
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
_PLAYER_STATIC = Path(__file__).resolve().parent / "static" / "player"
if _PLAYER_STATIC.exists():
    app.mount("/play-assets", StaticFiles(directory=_PLAYER_STATIC), name="play-assets")
_APP_STATIC = Path(__file__).resolve().parent / "static" / "app"
if _APP_STATIC.exists():
    app.mount("/app-assets", StaticFiles(directory=_APP_STATIC), name="app-assets")
_CHALLENGE_STATIC = Path(__file__).resolve().parent / "static" / "challenge"
if _CHALLENGE_STATIC.exists():
    app.mount(
        "/challenge-assets",
        StaticFiles(directory=_CHALLENGE_STATIC),
        name="challenge-assets",
    )
_CHALLENGE_CARD_FACES = (
    ROOT / "assets" / "cards" / "packs" / "goodall_modern_baloot_32" / "svg"
)
if _CHALLENGE_CARD_FACES.exists():
    app.mount(
        "/card-assets",
        StaticFiles(directory=_CHALLENGE_CARD_FACES),
        name="canonical-card-assets",
    )
    app.mount(
        "/challenge-card-faces",
        StaticFiles(directory=_CHALLENGE_CARD_FACES),
        name="challenge-card-faces",
    )
_CHALLENGE_VIDEO_ASSETS = ROOT / "assets" / "video" / "professional"
if _CHALLENGE_VIDEO_ASSETS.exists():
    app.mount(
        "/challenge-video-assets",
        StaticFiles(directory=_CHALLENGE_VIDEO_ASSETS),
        name="challenge-video-assets",
    )


@app.get("/favicon.ico", include_in_schema=False)
def application_favicon():
    """Use the governed professional card back as the shared app icon."""
    favicon = _CHALLENGE_VIDEO_ASSETS / "card-back.svg"
    if not favicon.is_file():
        raise HTTPException(404, "application icon is unavailable")
    return FileResponse(favicon, media_type="image/svg+xml")


_FOUNDRY_STATIC = Path(__file__).resolve().parent / "static" / "foundry"
if _FOUNDRY_STATIC.exists():
    app.mount(
        "/foundry-assets", StaticFiles(directory=_FOUNDRY_STATIC), name="foundry-assets"
    )
_GUIDE_STATIC = Path(__file__).resolve().parent / "static" / "guide"
if _GUIDE_STATIC.exists():
    app.mount(
        "/guide-assets", StaticFiles(directory=_GUIDE_STATIC), name="guide-assets"
    )
_FONT_STATIC = ROOT / "assets" / "fonts"
if _FONT_STATIC.exists():
    app.mount(
        "/foundry-fonts", StaticFiles(directory=_FONT_STATIC), name="foundry-fonts"
    )


@app.get("/configuration-studio", include_in_schema=False)
def configuration_studio():
    return RedirectResponse("/foundry?section=settings", status_code=307)


@app.get("/app", include_in_schema=False)
def unified_app():
    if not _APP_STATIC.exists():
        raise HTTPException(404, "application assets are unavailable")
    return FileResponse(_APP_STATIC / "index.html")


app.include_router(build_auth_router(IDENTITY))
app.include_router(build_sakka_router(SAKKA, IDENTITY))
app.include_router(build_shared_router(SAKKA, IDENTITY, EXPLANATIONS, REVIEWS))
app.include_router(build_live_router(LIVE, IDENTITY, GAMES, WEBSOCKET_TICKETS))
app.include_router(
    build_adjudication_router(ADJUDICATION, IDENTITY, AUTHORIZATION_POLICY)
)
app.include_router(
    build_hitl_router(
        HITL,
        IDENTITY,
        AUTHORIZATION_POLICY,
        HITL_REVIEW_RENDERER,
        OUTPUTS,
        HITL_PROFILE,
    )
)
app.include_router(
    build_configuration_router(
        CONFIGURATION,
        IDENTITY,
        AUTHORIZATION_POLICY,
        CONFIGURATION_PROFILE,
        PERFORMANCE_PROFILE,
    )
)
app.include_router(
    build_platform_v19_router(
        identities=IDENTITY,
        policy=AUTHORIZATION_POLICY,
        registry=REGISTRY,
        performance=PERFORMANCE,
        configuration=CONFIGURATION,
        capability_matrix=CAPABILITY_MATRIX,
        certification_profile=CERTIFICATION_PROFILE,
        benchmark_profile=BENCHMARK_PROFILE,
        operations_profile=OPERATIONS_PROFILE,
    )
)
app.include_router(
    build_security_router(
        IDENTITY,
        ACCESS_TOKENS,
        AUTHORIZATION_POLICY,
        OIDC_TRANSACTIONS,
        OIDC_PROVIDERS,
        OIDC_CALLBACK,
        PASSKEYS,
        WEBSOCKET_TICKETS,
        SECURITY_STATE,
    )
)
app.include_router(build_canonical_history_router(CANONICAL_HISTORY, IDENTITY, GAMES))


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _trusted_authorized(request: Request) -> bool:
    """Return whether this request may access complete private hands.

    Production deployments are deny-by-default. Local development and tests may
    opt in explicitly or rely on the loopback-only development default.
    """
    configured = os.environ.get("BALOOT_TRUSTED_API_KEY", "")
    supplied = request.headers.get("x-baloot-trusted-key", "")
    if configured:
        return bool(supplied) and hmac.compare_digest(configured, supplied)
    if _truthy("BALOOT_ALLOW_TRUSTED_CONTEXT"):
        return True
    environment = os.environ.get("BALOOT_ENV", "development").strip().lower()
    host = request.client.host if request.client is not None else ""
    return environment in {"development", "test"} and host in {
        "127.0.0.1",
        "::1",
        "localhost",
        "testclient",
    }


def _require_trusted(request: Request) -> None:
    if not _trusted_authorized(request):
        raise HTTPException(403, "trusted hidden-information access is not enabled")


_RESEARCH_MUTATION_OPENAPI = {
    "parameters": [
        {
            "name": "Idempotency-Key",
            "in": "header",
            "required": True,
            "description": (
                "Stable caller-supplied key. Reuse the same key only for the "
                "exact same Research recipe mutation intent."
            ),
            "schema": {
                "type": "string",
                "minLength": 1,
                "maxLength": IDEMPOTENCY_KEY_MAX_LENGTH,
                "pattern": IDEMPOTENCY_KEY_PATTERN,
            },
        }
    ]
}

_DURABLE_JOB_MUTATION_OPENAPI = {
    "parameters": [
        {
            "name": "Idempotency-Key",
            "in": "header",
            "required": True,
            "description": (
                "Stable caller-supplied key. Surrounding whitespace is "
                "removed before the key is used."
            ),
            "schema": {
                "type": "string",
                "minLength": 1,
                "pattern": r".*\S.*",
            },
        }
    ]
}

_RENDER_BATCH_MUTATION_OPENAPI = {
    "parameters": [
        {
            "name": "Idempotency-Key",
            "in": "header",
            "required": True,
            "description": (
                "Stable caller-supplied key containing at most 200 characters "
                "after surrounding whitespace is removed."
            ),
            "schema": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "pattern": r".*\S.*",
            },
        }
    ]
}


def _require_research_mutation_key(request: Request) -> str:
    value = request.headers.get("idempotency-key")
    if value is None or value == "":
        raise HTTPException(428, "Idempotency-Key header is required")
    try:
        return validate_idempotency_key(value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _require_trimmed_idempotency_key(
    request: Request,
    *,
    maximum_length: int | None = None,
) -> str:
    value = request.headers.get("idempotency-key", "").strip()
    if not value:
        raise HTTPException(428, "Idempotency-Key header is required")
    if maximum_length is not None and len(value) > maximum_length:
        raise HTTPException(
            422,
            f"Idempotency-Key must not exceed {maximum_length} characters",
        )
    return value


app.include_router(
    build_foundry_admin_v2_router(
        campaign_control=FOUNDRY_CAMPAIGNS,
        learning_store=LEARNING_STORE,
        learning=LEARNING,
        hitl=HITL,
        configuration=CONFIGURATION,
        registry=REGISTRY,
        foundry_catalogs=FOUNDRY_CATALOGS,
        context_searches=CONTEXT_SEARCHES,
        sakka_generations=SAKKA_GENERATIONS,
        identities=IDENTITY,
        require_trusted=_require_trusted,
        publication_policy=_CHALLENGE_PUBLICATION_POLICY,
        research=RESEARCH,
        render_batches=RENDER_BATCHES,
        active_bundle_hash=ACTIVE_BUNDLE.sha256,
        laffa_library=LAFFA_LIBRARY,
    )
)


def _client_key(request: Request) -> str:
    if _truthy("BALOOT_TRUST_PROXY"):
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client is not None else "unknown"


def _rate_allowed(request: Request) -> tuple[bool, int]:
    configured = os.environ.get("BALOOT_RATE_LIMIT_PER_MINUTE")
    environment = os.environ.get("BALOOT_ENV", "development").strip().lower()
    if configured is None and environment in {"development", "test"}:
        return True, 0
    limit = int(configured or "120")
    if (
        limit <= 0
        or request.url.path in {"/healthz", "/readyz", "/metrics"}
        or request.url.path.startswith("/outputs/")
    ):
        return True, 0
    now = time.monotonic()
    key = _client_key(request)
    with _RATE_LOCK:
        events = _RATE_EVENTS[key]
        while events and events[0] <= now - 60.0:
            events.popleft()
        if len(events) >= limit:
            retry = max(1, int(60.0 - (now - events[0])))
            return False, retry
        events.append(now)
    return True, 0


@app.middleware("http")
async def platform_security(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    maximum = int(os.environ.get("BALOOT_MAX_BODY_BYTES", "2097152"))
    length = request.headers.get("content-length")
    try:
        too_large = bool(length) and int(length) > maximum
    except ValueError:
        too_large = True
    if too_large:
        return JSONResponse(
            {"detail": "request body too large", "requestId": request_id},
            status_code=413,
        )
    allowed, retry_after = _rate_allowed(request)
    if not allowed:
        response = JSONResponse(
            {"detail": "rate limit exceeded", "requestId": request_id}, status_code=429
        )
        response.headers["retry-after"] = str(retry_after)
        return response
    token = request_token(request)
    identity, identity_session = IDENTITY.resolve(token)
    access_claims = None
    if identity is None and token and token.count(".") == 2:
        try:
            access_claims = ACCESS_TOKENS.verify(token)
        except (ValueError, KeyError, TypeError):
            access_claims = None
    request.state.identity = identity
    request.state.identity_session = identity_session
    request.state.access_claims = access_claims
    api_key = os.environ.get("BALOOT_API_KEY")
    supplied_key = request.headers.get("x-api-key", "")
    api_key_valid = (
        bool(api_key)
        and bool(supplied_key)
        and hmac.compare_digest(api_key, supplied_key)
    )
    request.state.api_key_valid = api_key_valid
    session_valid = (
        identity is not None and identity_session is not None
    ) or access_claims is not None
    auth_bootstrap = request.url.path.startswith("/api/v1/auth/")
    if (
        api_key
        and request.method not in {"GET", "HEAD", "OPTIONS"}
        and not (api_key_valid or session_valid or auth_bootstrap)
    ):
        return JSONResponse(
            {"detail": "authentication required", "requestId": request_id},
            status_code=401,
        )
    enforce_csrf = _truthy("BALOOT_ENFORCE_SESSION_CSRF") or (
        os.environ.get("BALOOT_ENV", "development").lower() == "production"
        and bool(IDENTITY_PROFILE.get("csrf", {}).get("enabled_in_production", True))
    )
    if (
        request.method not in {"GET", "HEAD", "OPTIONS"}
        and session_valid
        and not api_key_valid
        and access_claims is None
        and not auth_bootstrap
        and enforce_csrf
    ):
        supplied_csrf = request.headers.get("x-csrf-token", "")
        if not supplied_csrf or not hmac.compare_digest(
            str(identity_session["csrf_token"]), supplied_csrf
        ):
            return JSONResponse(
                {"detail": "csrf token required", "requestId": request_id},
                status_code=403,
            )
    claims_subject = str(
        getattr(access_claims, "subject_id", "")
        or (identity.subject_id if identity else "")
    )
    tenant = str(
        getattr(access_claims, "tenant_id", "")
        or request.headers.get("x-baloot-tenant-id", "")
    )
    role = str(
        getattr(access_claims, "role", "")
        or request.headers.get(
            "x-baloot-role", "player" if claims_subject else "anonymous"
        )
    )
    context_token = CURRENT_DATABASE_IDENTITY.set(
        DatabaseIdentityContext(
            subject_id=claims_subject,
            tenant_id=tenant,
            role=role,
            request_id=request_id,
            session_id=str(
                getattr(access_claims, "session_id", "")
                or (identity_session or {}).get("session_id", "")
            ),
        )
    )
    ownership_token = SERVICE_SUBJECT_OWNER.set(
        service_subject_ownership(request.headers, configured_key=api_key)
        if api_key_valid and not session_valid else None
    )
    started = time.monotonic()
    try:
        response = await call_next(request)
    finally:
        SERVICE_SUBJECT_OWNER.reset(ownership_token)
        CURRENT_DATABASE_IDENTITY.reset(context_token)
    elapsed = time.monotonic() - started
    with _METRIC_LOCK:
        _METRICS[(request.method, request.url.path, response.status_code)] += 1
        # Keep a coarse latency sum in integer microseconds without another dependency.
        _METRICS[("LATENCY_US", request.url.path, 0)] += int(elapsed * 1_000_000)
    response.headers["x-request-id"] = request_id
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["x-frame-options"] = "DENY"
    response.headers["referrer-policy"] = "no-referrer"
    response.headers["permissions-policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["content-security-policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    )
    return response


class VideoJobRequest(BaseModel):
    """Validated configuration for one round or full-match render job."""

    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    bundleRevisionId: str | None = Field(None, min_length=1, max_length=200)
    scope: Literal["round", "match"] = "round"
    maximumRounds: int = Field(16, ge=1, le=64)
    seed: int = Field(917, ge=0)
    rulesProfile: Literal["full", "sun_only", "hokum_only"] = "full"
    dealer: Literal["north", "west", "south", "east"] = "east"
    cutMode: Literal["auto", "none", "fixed"] = "auto"
    cutPosition: int = Field(12, ge=1, le=31)
    contract: Literal["auto", "sun", "hokum", "ashkal"] = "hokum"
    trumpSuit: Literal["auto", "spades", "hearts", "diamonds", "clubs"] = "auto"
    buyer: Literal["auto", "north", "west", "south", "east"] = "auto"
    firstLeaderPolicy: Literal["dealer_right", "buyer", "configured"] = "dealer_right"
    configuredFirstLeader: Literal["north", "west", "south", "east"] = "north"
    projectsEnabled: bool = True
    balootEnabled: bool = True
    ashkalEnabled: bool = True
    kaboutEnabled: bool = True
    mustTrumpWhenVoid: bool = True
    mustOvertrumpWhenPossible: bool = True
    allowDiscardWhenCannotOvertrump: bool = True
    partnerWinningException: Literal[
        "position_aware_ace_or_ekka", "always_allow_discard", "always_force_trump"
    ] = "position_aware_ace_or_ekka"
    matchTarget: int = Field(152, ge=1, le=10000)
    matchScoreUs: int = Field(0, ge=0)
    matchScoreThem: int = Field(0, ge=0)
    escalationLevel: Literal["normal", "double", "triple", "four", "gahwa"] = "normal"
    escalationMode: Literal["open", "locked"] = "open"
    requireProject: bool = True
    pauseTrickMin: int = Field(3, ge=0, le=7)
    pauseTrickMax: int = Field(6, ge=0, le=7)
    samples: int = Field(24, ge=2, le=1000)
    exactAllocationLimit: int = Field(4000, ge=1, le=500000)
    exactSearchRemaining: int = Field(8, ge=0, le=20)
    maxSeedAttempts: int = Field(120, ge=1, le=2000)
    maxCandidatesPerSeed: int = Field(1, ge=1, le=12)
    minimumHiddenDeals: int = Field(8, ge=1, le=1000000)
    minimumExpectedMarginGap: float = Field(0.5, ge=0.0, le=1000.0)
    minimumWinProbabilityGap: float = Field(0.01, ge=0.0, le=1.0)
    requireExactAnalysis: bool = False
    targetDebateScore: int = Field(65, ge=1, le=100)
    conventionProfile: Literal[
        "hard_facts_only", "experimental_display", "experimental_weighting"
    ] = "hard_facts_only"
    rendererPlugin: str = "premium_video"
    cardPack: str = "goodall_modern_baloot_32"
    cardBack: str = "1B"
    seatCallouts: bool = True
    voiceCallouts: bool = False
    tableFoley: bool = True
    spokenVoicePack: bool = False
    theme: Literal["emerald", "noir", "royal"] = "emerald"
    pacing: Literal["fast", "broadcast", "authentic", "cinematic"] = "authentic"
    cardOrientation: Literal["upright", "table"] = "upright"
    width: Literal[720, 1080] = 1080
    height: Literal[1280, 1920] = 1920
    fps: Literal[24, 30, 60] = 60
    crf: int = Field(19, ge=14, le=30)
    pauseSeconds: float = Field(6.5, ge=2.0, le=15.0)
    audio: bool = True
    titleAr: str = Field(_VIDEO_JOB_TITLE_DEFAULT, min_length=2, max_length=80)
    questionAr: str = Field(_VIDEO_JOB_QUESTION_DEFAULT, min_length=2, max_length=100)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "VideoJobRequest":
        if self.pauseTrickMin > self.pauseTrickMax:
            raise ValueError("pauseTrickMin must not exceed pauseTrickMax")
        if (self.width, self.height) not in {(720, 1280), (1080, 1920)}:
            raise ValueError("video resolution must be 720x1280 or 1080x1920")
        if self.rulesProfile == "sun_only" and self.contract not in {"auto", "sun"}:
            raise ValueError("sun_only permits only auto or sun")
        if self.rulesProfile == "hokum_only" and self.contract not in {"auto", "hokum"}:
            raise ValueError("hokum_only permits only auto or hokum")
        if not self.ashkalEnabled and self.contract == "ashkal":
            raise ValueError("Ashkal is disabled")
        if self.contract != "hokum" and self.trumpSuit != "auto":
            raise ValueError("trumpSuit is only applicable to Hokum")
        if self.contract == "ashkal" and self.buyer != "auto":
            raise ValueError(
                "Ashkal purchase recipient is derived from the bidder partnership"
            )
        if self.contract in {"sun", "ashkal"} and self.escalationLevel not in {
            "normal",
            "double",
        }:
            raise ValueError(
                "Sun-like contracts permit only normal or double in this profile"
            )
        if (
            self.escalationLevel not in {"double", "four"}
            and self.escalationMode != "open"
        ):
            raise ValueError("locked mode is only meaningful for double or four")
        if not self.projectsEnabled:
            # Friendly normalization for partial API/Studio requests: disabling
            # projects also disables dependent Baloot and project filters.
            self.balootEnabled = False
            self.requireProject = False
        CARD_PACKS.get(self.cardPack).validate_back(self.cardBack)
        return self


class GameCreateRequest(BaseModel):
    seed: int = 1
    dealer: str = "east"
    rulesProfile: str = "full"
    matchScoreUs: int = 0
    matchScoreThem: int = 0


class BidRequest(BaseModel):
    seat: str
    action: str
    trumpSuit: str | None = None
    expectedVersion: int | None = None


class PlayRequest(BaseModel):
    seat: str
    card: str | int | dict
    ekka: bool = False
    expectedVersion: int | None = None


class ContextRequest(BaseModel):
    state: dict[str, Any]
    rulesProfile: str = "full"


class AnnulRequest(BaseModel):
    seat: str
    reason: str = "kawesh"
    expectedVersion: int | None = None


class AnalysisRequest(BaseModel):
    state: dict[str, Any]
    observer: str = "south"
    rulesProfile: str = "full"
    searchMethod: Literal[
        "paired_hidden_deals", "information_set_mcts", "hybrid_team_public"
    ] = "hybrid_team_public"
    samples: int = Field(80, ge=2, le=BELIEF_POLICY.maximum_particles)
    exactAllocationLimit: int = Field(
        20000,
        ge=1,
        le=BELIEF_POLICY.maximum_exact_allocation_limit,
    )
    exactSearchRemaining: int = Field(12, ge=0, le=24)
    ismctsIterations: int = Field(2000, ge=16, le=1000000)
    ismctsParticles: int = Field(
        256,
        ge=4,
        le=BELIEF_POLICY.maximum_particles,
    )
    ismctsExploration: float = Field(1.15, ge=0.0, le=10.0)
    useExperimentalConventions: bool = False
    seed: int = 1


class ReanalysisRequest(BaseModel):
    samples: int = Field(96, ge=2, le=BELIEF_POLICY.maximum_particles)
    exactAllocationLimit: int = Field(
        20000,
        ge=1,
        le=BELIEF_POLICY.maximum_exact_allocation_limit,
    )
    exactSearchRemaining: int = Field(10, ge=0, le=24)
    useExperimentalConventions: bool = False
    seed: int = 1


class ReplayAuditRequest(BaseModel):
    mode: Literal["observer_safe", "trusted_omniscient", "consensus"] = "observer_safe"
    rerun: bool = False
    maximumDecisions: int = Field(12, ge=1, le=1000)
    samples: int = Field(96, ge=2, le=BELIEF_POLICY.maximum_particles)
    exactAllocationLimit: int = Field(
        20000,
        ge=1,
        le=BELIEF_POLICY.maximum_exact_allocation_limit,
    )
    exactSearchRemaining: int = Field(10, ge=0, le=24)
    useExperimentalConventions: bool = False
    seed: int = 1


class DecisionAuditRequest(BaseModel):
    analysisId: str | None = None
    rerun: bool = False
    samples: int = Field(96, ge=2, le=BELIEF_POLICY.maximum_particles)
    exactAllocationLimit: int = Field(
        20000,
        ge=1,
        le=BELIEF_POLICY.maximum_exact_allocation_limit,
    )
    exactSearchRemaining: int = Field(10, ge=0, le=24)
    useExperimentalConventions: bool = False
    seed: int = 1


class DifficultyAttemptRequest(BaseModel):
    playerId: str | None = Field(None, max_length=200)
    chosenAction: Any
    responseSeconds: float | None = Field(None, ge=0.0, le=3600.0)
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    reportedSkill: float | None = Field(None, ge=0.0, le=100.0)
    changedAnswer: bool = False


class FeedbackRequest(BaseModel):
    preferredAction: Any
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    rationale: str = Field("", max_length=4000)
    source: Literal["player", "expert", "referee", "audience", "engine", "model"] = (
        "expert"
    )
    reviewer: str | None = Field(None, max_length=200)


class ForkDecisionRequest(BaseModel):
    action: Any | None = None
    continueToEnd: bool = False
    maximumActions: int = Field(128, ge=1, le=10000)


class DatasetBuildRequest(BaseModel):
    purpose: str = Field("human_corrected_play_v1", min_length=1, max_length=200)
    minimumConfidence: float | None = Field(None, ge=0.0, le=1.0)


class ModelTrainRequest(BaseModel):
    datasetId: str | None = None
    backend: Literal["auto", "numpy", "torch"] = "auto"
    epochs: int | None = Field(None, ge=1, le=10000)
    learningRate: float | None = Field(None, gt=0.0, le=10.0)
    l2: float | None = Field(None, ge=0.0, le=10.0)
    batchSize: int | None = Field(None, ge=1, le=100000)
    seed: int | None = None


class ModelEvaluateRequest(BaseModel):
    datasetId: str | None = None


class ModelPromotionRequest(BaseModel):
    rollbackModelId: str = Field(min_length=1, max_length=200)
    decisionCount: int = Field(ge=1)
    boardSets: int = Field(ge=1)
    scoreRate: float = Field(ge=0.0, le=1.0)
    scoreRateLowerBound: float = Field(ge=0.0, le=1.0)
    calibrationError: float = Field(ge=0.0, le=1.0)
    p95LatencyMs: float = Field(ge=0.0)
    oodRecall: float = Field(ge=0.0, le=1.0)
    legalViolationRate: float = Field(ge=0.0, le=1.0)


class ChallengeMiningRequest(BaseModel):
    source: Literal["simulation", "history", "both"] = str(
        _CHALLENGE_REQUEST_DEFAULTS["source"]
    )
    decisionKind: Literal["play", "bid", "any"] = str(
        _CHALLENGE_REQUEST_DEFAULTS["decision_kind"]
    )
    seedStart: int = int(_CHALLENGE_REQUEST_DEFAULTS["seed_start"])
    seeds: int = Field(
        int(_CHALLENGE_REQUEST_DEFAULTS["seeds"]),
        ge=1,
        le=1000000,
    )
    contract: Literal["any", "sun", "hokum", "ashkal"] = str(
        _CHALLENGE_REQUEST_DEFAULTS["contract"]
    )
    buyerRelationship: Literal["any", "self", "partner", "opponent"] = str(
        _CHALLENGE_REQUEST_DEFAULTS["buyer_relationship"]
    )
    archetype: str = str(_CHALLENGE_REQUEST_DEFAULTS["archetype"])
    difficultyMin: int = Field(
        int(_CHALLENGE_REQUEST_DEFAULTS["difficulty_min"]),
        ge=0,
        le=100,
    )
    difficultyMax: int = Field(
        int(_CHALLENGE_REQUEST_DEFAULTS["difficulty_max"]),
        ge=0,
        le=100,
    )
    minimumPublishability: float = Field(
        float(_CHALLENGE_REQUEST_DEFAULTS["minimum_publishability"]),
        ge=0.0,
        le=1.0,
    )
    maximumResults: int = Field(
        int(_CHALLENGE_REQUEST_DEFAULTS["maximum_results"]),
        ge=1,
        le=500,
    )
    requireProject: bool = bool(_CHALLENGE_REQUEST_DEFAULTS["require_project"])
    pauseTrickMin: int = Field(
        int(_CHALLENGE_REQUEST_DEFAULTS["pause_trick_min"]),
        ge=0,
        le=7,
    )
    pauseTrickMax: int = Field(
        int(_CHALLENGE_REQUEST_DEFAULTS["pause_trick_max"]),
        ge=0,
        le=7,
    )
    maximumSeconds: float = Field(
        float(_CHALLENGE_REQUEST_DEFAULTS["maximum_seconds"]),
        gt=0.0,
        le=86400.0,
    )
    shallowSamples: int | None = Field(
        None,
        ge=2,
        le=BELIEF_POLICY.maximum_particles,
    )
    deepSamples: int | None = Field(
        None,
        ge=2,
        le=BELIEF_POLICY.maximum_particles,
    )
    exactAllocationLimit: int | None = Field(
        None,
        ge=1,
        le=BELIEF_POLICY.maximum_exact_allocation_limit,
    )
    exactSearchRemaining: int | None = Field(None, ge=0, le=24)
    playerProfiles: dict[str, str] = Field(
        default=dict(_CHALLENGE_REQUEST_DEFAULTS["player_profiles"])
    )
    contentContract: str = str(_CHALLENGE_REQUEST_DEFAULTS["content_contract"])

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.difficultyMin > self.difficultyMax:
            raise ValueError("difficultyMin cannot exceed difficultyMax")
        if self.pauseTrickMin > self.pauseTrickMax:
            raise ValueError("pauseTrickMin cannot exceed pauseTrickMax")
        if self.contentContract not in _CHALLENGE_CONTENT_CONTRACTS:
            raise ValueError("contentContract is not governed")
        return self


class ResearchOutputRequest(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal["laffa", "laffa_collection", "contextualized_laffa", "sakka"] = (
        "laffa"
    )
    resultCount: int = Field(1, ge=1, le=1000)


class ResearchOutputV3Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    kind: Literal["laffa", "laffa_collection"] = "laffa"
    resultCount: int = Field(1, ge=1, le=1000)


class ResearchProfilesRequest(BaseModel):
    model_config = {"extra": "forbid"}
    domainHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    conventionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agentRelease: str = Field(min_length=1, max_length=200)
    analysisProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scoreProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchProjectConstraintRequest(BaseModel):
    model_config = {"extra": "forbid"}
    seat: Literal["north", "west", "south", "east"]
    contract: Literal["sun", "hokum", "ashkal"]
    kind: Literal[
        "siri", "fifty", "hundred_sequence", "hundred_four_kind", "four_hundred"
    ]
    value: int = Field(ge=0, le=400)
    highRank: Literal["seven", "eight", "nine", "ten", "jack", "queen", "king", "ace"]
    mode: Literal["hard", "preference", "qualification_gate"] = "hard"


class ResearchProjectConstraintV3Request(ResearchProjectConstraintRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    mode: Literal["hard"] = "hard"


class ResearchSituationRequest(BaseModel):
    model_config = {"extra": "forbid"}
    dealer: Literal["north", "west", "south", "east"] = "east"
    buyer: Literal["north", "west", "south", "east"] | None = None
    objectiveSeat: Literal["north", "west", "south", "east"] | None = None
    constructionBuyerRelation: Literal["automatic", "objective", "partner"] = (
        "automatic"
    )
    purchaseRound: Literal[1, 2] | None = None
    purchaseSuits: list[Literal["spades", "hearts", "diamonds", "clubs"]] = Field(
        default_factory=list
    )
    trumpSuit: Literal["spades", "hearts", "diamonds", "clubs"] | None = None
    contracts: list[Literal["sun", "hokum", "ashkal"]] = Field(default_factory=list)
    matchScore: list[Annotated[int, Field(ge=0, le=10_000)]] = Field(
        default_factory=lambda: [0, 0], min_length=2, max_length=2
    )
    requiredCards: dict[str, list[str]] = Field(default_factory=dict)
    forbiddenCards: dict[str, list[str]] = Field(default_factory=dict)
    finalSuitCountRanges: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict
    )
    projects: list[ResearchProjectConstraintRequest] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list)


ResearchSeatName = Literal["north", "west", "south", "east"]
ResearchSuitName = Literal["spades", "hearts", "diamonds", "clubs"]
ResearchCardCode = Annotated[str, Field(pattern=r"^[SHDC](?:10|[789JQKA])$")]


class ResearchSeatCardConstraintsV3Request(
    RootModel[dict[ResearchSeatName, list[ResearchCardCode]]]
):
    model_config = {"strict": True}


class ResearchSuitRangeV3Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    minimum: int = Field(0, ge=0, le=8)
    maximum: int = Field(8, ge=0, le=8)


class ResearchSeatSuitRangesV3Request(
    RootModel[
        dict[
            ResearchSeatName,
            dict[ResearchSuitName, ResearchSuitRangeV3Request],
        ]
    ]
):
    model_config = {"strict": True}


class ResearchSituationV3Request(ResearchSituationRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    contracts: list[Literal["sun", "hokum", "ashkal"]] = Field(
        default_factory=lambda: ["sun"]
    )
    requiredCards: ResearchSeatCardConstraintsV3Request = Field(
        default_factory=lambda: ResearchSeatCardConstraintsV3Request({})
    )
    forbiddenCards: ResearchSeatCardConstraintsV3Request = Field(
        default_factory=lambda: ResearchSeatCardConstraintsV3Request({})
    )
    finalSuitCountRanges: ResearchSeatSuitRangesV3Request = Field(
        default_factory=lambda: ResearchSeatSuitRangesV3Request({})
    )
    projects: list[ResearchProjectConstraintV3Request] = Field(default_factory=list)
    escalations: list[Literal["normal", "double", "triple", "four", "gahwa"]] = Field(
        default_factory=list, max_length=1
    )


class ResearchTacticRequest(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal[
        "only_winning_move",
        "small_card_big_gain",
        "large_card_sacrifice",
        "kabout_creation",
        "kabout_preservation",
        "kabout_prevention",
        "contract_rescue",
        "entry_unblocking",
        "trump_control",
        "partner_enablement",
        "project_bidding",
        "debate",
    ]
    mode: Literal["hard", "preference", "qualification_gate"] = "preference"
    minimumScore: float = Field(0.0, ge=0.0, le=100.0)
    weight: float = Field(1.0, gt=0.0, le=100.0)
    perspectives: list[Literal["north", "west", "south", "east"]] = Field(
        default_factory=list
    )


class ResearchTacticV3Request(ResearchTacticRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    mode: Literal["preference", "qualification_gate"] = "preference"


class ResearchQualificationRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    minimumTruth: Literal[
        "exact_optimal", "robust_best_known", "debatable", "provisional"
    ] = "robust_best_known"
    minimumConfidence: float = Field(0.75, ge=0.0, le=1.0)
    minimumSurroundingQuality: float = Field(70.0, ge=0.0, le=100.0)
    requireObserverSafety: Literal[True] = True
    requireAllDecisionAudit: Literal[True] = True
    maximumRegret: float = Field(0.0, ge=0.0)


class ResearchConstructionPreScreenRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    id: str = Field(min_length=1, max_length=120)
    revision: str = Field(min_length=1, max_length=40)
    enabled: bool = False
    maximumPlayDecisions: int = Field(4, ge=1, le=16)
    maximumBidRegret: float = Field(0.0, ge=0.0)
    maximumPlayRegret: float = Field(0.0, ge=0.0)
    maximumProbabilityDrop: float = Field(0.0, ge=0.0, le=1.0)
    minimumConfidence: float = Field(0.0, ge=0.0, le=1.0)
    rejectCatastrophicCrossing: bool = True
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchTacticalViabilityPreScreenRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    id: Literal["exact_tactical_viability_screen_v1"] = (
        "exact_tactical_viability_screen_v1"
    )
    revision: str = Field("1", pattern=r"^[1-9][0-9]*$")
    enabled: bool = False
    maximumDecisions: int = Field(8, ge=1, le=16)
    requiredTacticsOnly: bool = True
    rejectWhenExactAbsent: bool = True
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchTacticalConstructionRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    id: Literal["tactical_play_precursor_v1"] = "tactical_play_precursor_v1"
    revision: int = Field(1, ge=1)
    enabled: bool = True
    maximumPayoffTricks: int = Field(4, ge=1, le=4)
    minimumLaterRawPoints: int = Field(10, ge=1, le=40)
    requirePrecursorForGatedTactics: bool = True
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchTacticalDealBalanceRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    id: Literal["balanced_tactical_deal_v1"] = "balanced_tactical_deal_v1"
    revision: int = Field(1, ge=1)
    enabled: bool = True
    maximumTeamHighCardDifference: int = Field(2, ge=0, le=12)
    minimumHighCardsPerSeat: int = Field(2, ge=0, le=3)
    maximumHighCardsPerSeat: int = Field(4, ge=3, le=8)
    minimumTargetLowCards: int = Field(1, ge=0, le=3)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchStrategicPlayRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    minimumPlayedCards: int = Field(24, ge=20, le=28)
    requestedSamples: int = Field(32, ge=1, le=5_000)
    exactAllocationLimit: int = Field(500, ge=1, le=5_000)
    maximumSamplingAttempts: int = Field(500, ge=1, le=1_000_000)
    maximumWorlds: int = Field(500, ge=1, le=5_000)
    maximumContinuationSteps: int = Field(20_000, ge=1, le=2_000_000)
    maximumPlies: int = Field(8, ge=4, le=12)
    maximumSelectionSeconds: float = Field(30.0, ge=0.01, le=30.0)
    requireExactForQualification: bool = True
    continuationPolicy: Literal["observer_safe_heuristic_v1"] = (
        "observer_safe_heuristic_v1"
    )
    policyProfileId: Literal["maximin_margin"] = "maximin_margin"
    policyRevision: Literal[1] = 1
    outcomeBoundary: int = Field(1, ge=-10_000, le=10_000)
    utility: Literal["actor_team_awarded_point_margin"] = (
        "actor_team_awarded_point_margin"
    )
    tieBreak: Literal["maximin_then_total_then_card_id"] = (
        "maximin_then_total_then_card_id"
    )
    solverBackend: Literal["batched_cp_sat"] | None = None
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchStrategicAuctionRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    requestedSamples: int = Field(16, ge=2, le=1_000)
    integerScale: int = Field(1_000, ge=100, le=100_000)
    maximumConflicts: int = Field(50_000, ge=1, le=10_000_000)
    maximumSelectionSeconds: float = Field(5.0, ge=0.01, le=30.0)
    objective: Literal["expected_round_margin"] = "expected_round_margin"
    tieBreak: Literal["win_probability_then_best_share_then_action_id"] = (
        "win_probability_then_best_share_then_action_id"
    )
    requireOptimalForQualification: bool = True
    solverBackend: Literal["ortools_cp_sat"] | None = None
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchStrategicPlayV3Request(ResearchStrategicPlayRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}


class ResearchStrategicAuctionV3Request(ResearchStrategicAuctionRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    requestedSamples: Literal[192] = Field(
        RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
        validate_default=True,
    )
    tieBreak: Literal[
        "round_lead_probability_then_best_share_then_legal_action_order_v1"
    ] = Field(
        RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
        validate_default=True,
    )


class ResearchResourcesRequest(BaseModel):
    model_config = {"extra": "forbid"}
    generationLane: Literal["ortools", "organic_runtime"] = "ortools"
    seedStart: int = Field(1, ge=0)
    seedCount: int = Field(10, ge=1, le=1_000_000_000)
    seedsPerShard: int = Field(10, ge=1, le=1_000_000)
    candidatesPerSeed: int = Field(1, ge=1, le=100)
    solverTimeoutSeconds: float = Field(15.0, ge=0.1, le=900.0)
    maximumConflicts: int = Field(250_000, ge=1, le=10_000_000)
    analysisMode: Literal["screening", "cascaded", "deep"] = "cascaded"
    deepVerificationPolicy: Literal["off", "required_gates", "all_candidates"] = (
        "required_gates"
    )
    constructionPreScreen: ResearchConstructionPreScreenRequest | None = None
    tacticalViabilityPreScreen: ResearchTacticalViabilityPreScreenRequest | None = None
    tacticalConstruction: ResearchTacticalConstructionRequest | None = None
    tacticalDealBalance: ResearchTacticalDealBalanceRequest | None = None
    auctionConstructionMode: (
        Literal[
            "actor_safe_agent_verified",
            "cp_sat_actor_safe_decision_v1",
            "cp_sat_feasibility_line",
        ]
        | None
    ) = None
    playConstructionMode: (
        Literal[
            "actor_safe_agent_verified",
            "cp_sat_actor_safe_late_game_v1",
            "cp_sat_feasibility_line",
        ]
        | None
    ) = None
    strategicPlay: ResearchStrategicPlayRequest | None = None
    strategicAuction: ResearchStrategicAuctionRequest | None = None


class ResearchConstructionPreScreenV3Request(ResearchConstructionPreScreenRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    id: Literal["actor_safe_pre_screen_v1"] = "actor_safe_pre_screen_v1"
    revision: Literal["1"] = "1"


class ResearchTacticalViabilityPreScreenV3Request(
    ResearchTacticalViabilityPreScreenRequest
):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    revision: Literal["1"] = "1"


class ResearchTacticalConstructionV3Request(ResearchTacticalConstructionRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    revision: Literal[1] = 1
    enabled: bool = False


class ResearchTacticalDealBalanceV3Request(ResearchTacticalDealBalanceRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    revision: Literal[1] = 1
    enabled: bool = False


class ResearchResourcesV3Request(ResearchResourcesRequest):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    candidatesPerSeed: int | None = Field(None, ge=1, le=100)
    solverTimeoutSeconds: float | None = Field(None, ge=0.1, le=900.0)
    maximumConflicts: int | None = Field(None, ge=1, le=10_000_000)
    constructionPreScreen: ResearchConstructionPreScreenV3Request | None = None
    tacticalViabilityPreScreen: ResearchTacticalViabilityPreScreenV3Request | None = (
        None
    )
    tacticalConstruction: ResearchTacticalConstructionV3Request | None = None
    tacticalDealBalance: ResearchTacticalDealBalanceV3Request | None = None
    strategicPlay: ResearchStrategicPlayV3Request | None = None
    strategicAuction: ResearchStrategicAuctionV3Request | None = None

    @model_validator(mode="after")
    def materialize_lane_defaults(self):
        for field_name in (
            "constructionPreScreen",
            "tacticalViabilityPreScreen",
            "tacticalConstruction",
            "tacticalDealBalance",
        ):
            profile = getattr(self, field_name)
            if profile is not None and profile.enabled is False:
                setattr(self, field_name, None)
        if self.generationLane == "ortools":
            if self.candidatesPerSeed is None:
                self.candidatesPerSeed = 1
            if self.solverTimeoutSeconds is None:
                self.solverTimeoutSeconds = 15.0
            if self.maximumConflicts is None:
                self.maximumConflicts = 250_000
            if self.auctionConstructionMode is None:
                self.auctionConstructionMode = "actor_safe_agent_verified"
            if self.playConstructionMode is None:
                self.playConstructionMode = "actor_safe_agent_verified"
        elif {
            "candidatesPerSeed",
            "solverTimeoutSeconds",
            "maximumConflicts",
        }.intersection(self.model_fields_set):
            raise ValueError("organic runtime does not accept OR-Tools solver controls")
        return self


class ResearchRecipeRequest(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1]
    code: str | None = Field(None, pattern=r"^RCP-[A-Z0-9]{6,20}$")
    titleAr: str = Field(min_length=3, max_length=160)
    output: ResearchOutputRequest
    profiles: ResearchProfilesRequest
    situation: ResearchSituationRequest
    tactics: list[ResearchTacticRequest] = Field(default_factory=list)
    qualification: ResearchQualificationRequest = Field(
        default_factory=ResearchQualificationRequest
    )
    resources: ResearchResourcesRequest = Field(
        default_factory=ResearchResourcesRequest
    )
    status: Literal["draft", "published", "retired"] = "draft"


class ResearchDiversityRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    schemaVersion: Literal[1] = 1
    cellDimensions: list[
        Literal[
            "decision_kind",
            "contract",
            "purchase_round",
            "trump_suit",
            "project",
            "escalation",
            "tactical_detector",
            "trick_number",
            "acting_seat",
            "buyer_relationship",
            "score_pressure",
            "difficulty_band",
            "perspective_breadth",
            "source",
            "video_duration",
            "surrounding_competence",
            "candidate_scope",
        ]
    ] = Field(
        default_factory=lambda: [
            "contract",
            "tactical_detector",
            "acting_seat",
            "source",
        ],
        min_length=1,
    )
    topKPerCell: int = Field(3, ge=1, le=1000)
    rejectedAuditModulus: int = Field(100, ge=1, le=1_000_000)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ResearchDiversityV3Request(ResearchDiversityRequest):
    cellDimensions: list[
        Literal[
            "contract",
            "purchase_round",
            "trump_suit",
            "project",
            "escalation",
            "tactical_detector",
            "acting_seat",
            "score_pressure",
            "perspective_breadth",
            "source",
            "candidate_scope",
        ]
    ] = Field(
        default_factory=lambda: [
            "contract",
            "tactical_detector",
            "acting_seat",
            "source",
        ],
        min_length=1,
    )


class ResearchRenderPolicyRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    mode: Literal["on_demand"] = "on_demand"


class ResearchRenderPolicyV4Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    mode: Literal["on_demand", "disabled"] = "on_demand"
    videoCount: int = Field(1, ge=0, le=100)

    @model_validator(mode="before")
    @classmethod
    def default_disabled_video_count(cls, value):
        if (
            isinstance(value, dict)
            and value.get("mode") == "disabled"
            and "videoCount" not in value
        ):
            return {**value, "videoCount": 0}
        return value


class ResearchRecipeV2Request(ResearchRecipeRequest):
    schemaVersion: Literal[2]
    diversity: ResearchDiversityRequest = Field(
        default_factory=ResearchDiversityRequest
    )
    renderPolicy: ResearchRenderPolicyRequest = Field(
        default_factory=ResearchRenderPolicyRequest
    )


class ResearchConfigurationSelectorRequest(BaseModel):
    """The only configuration choice accepted from a V3 recipe client."""

    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    schemaVersion: Literal[1] = 1
    bundleRevisionId: str = Field(min_length=1, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def validate_selector(cls, value):
        if isinstance(value, Mapping):
            return build_research_configuration_selector_v1({"schemaVersion": 1, **value})
        return value


class ResearchSnapshotArtifactResponse(BaseModel):
    model_config = {"extra": "forbid"}
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)
    mediaType: Literal["application/vnd.balootx.effective-runtime-config-v2+json"]
    name: str | None = None

    @model_validator(mode="after")
    def validate_content_addressed_uri(self):
        local_uri = f"cas://sha256/{self.sha256}"
        s3_pattern = rf"^s3://[^/?#]+/[^?#]+/{self.sha256}$"
        if self.uri != local_uri and re.fullmatch(s3_pattern, self.uri) is None:
            raise ValueError(
                "snapshot artifact URI must be governed content-addressed storage"
            )
        return self


class ResearchComponentRevisionIdentityResponse(BaseModel):
    model_config = {"extra": "forbid"}
    revisionId: str = Field(min_length=1)
    profileId: str = Field(min_length=1)
    version: str = Field(min_length=1)
    payloadSha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchFrozenSelectionIdentityResponse(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal[
        "domain",
        "rules",
        "convention",
        "belief",
        "agent",
        "analyzer",
        "score",
        "decisionScore",
    ]
    id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sourceRevisions: dict[str, str] = Field(min_length=1)


class ResearchFrozenSelectionIdentitiesResponse(BaseModel):
    model_config = {"extra": "forbid"}
    domain: ResearchFrozenSelectionIdentityResponse
    rules: ResearchFrozenSelectionIdentityResponse
    convention: ResearchFrozenSelectionIdentityResponse
    belief: ResearchFrozenSelectionIdentityResponse
    agent: ResearchFrozenSelectionIdentityResponse
    analyzer: ResearchFrozenSelectionIdentityResponse
    score: ResearchFrozenSelectionIdentityResponse
    decisionScore: ResearchFrozenSelectionIdentityResponse


class ResearchConfigurationAuthorityResponse(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    bundleRevisionId: str = Field(min_length=1, max_length=200)
    bundleId: str = Field(min_length=1, max_length=200)
    bundleVersion: str = Field(
        pattern=(
            r"^[0-9]+\.[0-9]+\.[0-9]+"
            r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
        )
    )
    semanticBundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveConfigHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveRuntimeConfigHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshotContract: Literal["EffectiveRuntimeConfigV2"]
    snapshotArtifact: ResearchSnapshotArtifactResponse
    componentRevisions: dict[str, ResearchComponentRevisionIdentityResponse]
    identities: ResearchFrozenSelectionIdentitiesResponse
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_sources(self):
        expected_sources = {
            "domain": {"rules", "conventions", "product"},
            "rules": {"rules"},
            "convention": {"conventions"},
            "belief": {"belief"},
            "agent": {"agents", "challenge"},
            "analyzer": {
                "challenge",
                "analysis",
                "belief",
                "match_value",
                "performance",
                "rules",
                "conventions",
            },
            "score": {"challenge"},
            "decisionScore": {"challenge"},
        }
        for role, expected_families in expected_sources.items():
            identity = getattr(self.identities, role)
            if identity.kind != role:
                raise ValueError(f"{role} identity kind mismatch")
            if set(identity.sourceRevisions) != expected_families:
                raise ValueError(f"{role} identity source revisions are incomplete")
            for family, revision_id in identity.sourceRevisions.items():
                component = self.componentRevisions.get(family)
                if component is None or component.revisionId != revision_id:
                    raise ValueError(f"{role} identity source revision mismatch")
        return self


class ResearchRecipeV3CreateRequest(BaseModel):
    """Client request; Phase 2 resolves and freezes every derived identity."""

    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    schemaVersion: Literal[3]
    code: str | None = Field(None, pattern=r"^RCP-[A-Z0-9]{6,20}$")
    titleAr: str = Field(min_length=3, max_length=160)
    output: ResearchOutputV3Request = Field(default_factory=ResearchOutputV3Request)
    configuration: ResearchConfigurationSelectorRequest
    situation: ResearchSituationV3Request = Field(
        default_factory=ResearchSituationV3Request
    )
    tactics: list[ResearchTacticV3Request] = Field(
        default_factory=lambda: [
            ResearchTacticV3Request(
                kind="small_card_big_gain",
                mode="qualification_gate",
                minimumScore=70.0,
                weight=1.0,
                perspectives=[],
            )
        ]
    )
    qualification: ResearchQualificationRequest = Field(
        default_factory=ResearchQualificationRequest
    )
    resources: ResearchResourcesV3Request = Field(
        default_factory=lambda: ResearchResourcesV3Request(
            auctionConstructionMode="actor_safe_agent_verified",
            playConstructionMode="actor_safe_agent_verified",
        )
    )
    diversity: ResearchDiversityV3Request = Field(
        default_factory=ResearchDiversityV3Request
    )
    renderPolicy: ResearchRenderPolicyRequest = Field(
        default_factory=ResearchRenderPolicyRequest
    )
    status: Literal["draft", "published", "retired"] = "published"


class ScenarioPredicateV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    predicateId: str = Field(pattern=r"^PRED-[A-Z0-9][A-Z0-9_-]{2,63}$")
    kind: str = Field(min_length=1, max_length=120)
    version: int = Field(ge=1)
    hardness: Literal["hard", "prefer", "report"]
    arguments: dict[str, Any]
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ScenarioExpressionV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    op: Literal["all", "any", "not", "count"] | None = None
    children: list["ScenarioExpressionV1Request"] | None = None
    minimum: int | None = Field(None, ge=0)
    maximum: int | None = Field(None, ge=0)
    predicateId: str | None = Field(None, pattern=r"^PRED-[A-Z0-9][A-Z0-9_-]{2,63}$")
    kind: str | None = Field(None, min_length=1, max_length=120)
    version: int | None = Field(None, ge=1)
    hardness: Literal["hard", "prefer", "report"] | None = None
    arguments: dict[str, Any] | None = None
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_expression_shape(self):
        predicate_fields = (
            self.predicateId,
            self.kind,
            self.version,
            self.hardness,
            self.arguments,
        )
        is_predicate = any(value is not None for value in predicate_fields)
        if is_predicate:
            if any(value is None for value in predicate_fields):
                raise ValueError("scenario predicate expression is incomplete")
            if self.op is not None or self.children is not None:
                raise ValueError(
                    "scenario expression cannot be both operator and predicate"
                )
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("scenario predicate cannot contain count bounds")
            return self
        if self.op is None or self.children is None:
            raise ValueError("scenario operator expression requires op and children")
        if not 1 <= len(self.children) <= 64:
            raise ValueError("scenario expression requires between 1 and 64 children")
        if self.op == "not" and len(self.children) != 1:
            raise ValueError("scenario not expression requires one child")
        if self.op == "count":
            if self.minimum is None or self.maximum is None:
                raise ValueError(
                    "scenario count expression requires minimum and maximum"
                )
            if self.minimum > self.maximum or self.maximum > len(self.children):
                raise ValueError("scenario count expression bounds are invalid")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError("count bounds are valid only for count expressions")
        return self


class ExactAuctionActionV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    actor: Literal["north", "west", "south", "east"] | None = None
    action: str = Field(min_length=1, max_length=40)
    trumpSuit: Literal["spades", "hearts", "diamonds", "clubs"] | None = None


class ExactEscalationActionV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    actor: Literal["north", "west", "south", "east"]
    action: str = Field(min_length=1, max_length=40)
    mode: str | None = Field(None, min_length=1, max_length=40)
    requestedMode: str | None = Field(None, min_length=1, max_length=40)

    @model_validator(mode="after")
    def validate_mode_alias(self):
        if self.mode is not None and self.requestedMode is not None:
            raise ValueError("exact escalation action cannot contain both mode aliases")
        return self


class ExactProjectAssertionV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    owner: Literal["north", "west", "south", "east"] | None = None
    team: Literal["north_south", "west_east"] | None = None
    kind: str | None = Field(None, min_length=1, max_length=80)
    value: int | None = Field(None, ge=0)
    highRank: int | None = Field(None, ge=0)
    state: Literal["eligible", "declared", "comparison_winner", "awarded"] = "eligible"


class ExactProjectDeclarationV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    owner: Literal["north", "west", "south", "east"]
    kind: str = Field(min_length=1, max_length=80)
    value: int = Field(ge=0)
    highRank: int = Field(ge=0)
    declarationOrder: int = Field(ge=1)


class ExactProjectActionV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    actor: Literal["north", "west", "south", "east"]
    action: Literal["pass", "declare"]
    projectId: str | None = Field(None, pattern=r"^PROJECT-[0-9A-F]{16}$")

    @model_validator(mode="after")
    def validate_project_action(self):
        if self.action == "pass" and self.projectId is not None:
            raise ValueError("project pass cannot contain projectId")
        if self.action == "declare" and self.projectId is None:
            raise ValueError("project declare requires projectId")
        return self


class ExactPlayActionV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    actor: Literal["north", "west", "south", "east"] | None = None
    card: int | str
    ekka: bool | None = None


class ExactAdjudicationEventV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    kind: str = Field(min_length=1, max_length=120)
    actionSequence: int = Field(ge=0)
    actorSeat: Literal["north", "west", "south", "east"]
    expectedActorSeat: Literal["north", "west", "south", "east"] | None = None
    attemptedAction: dict[str, str | int | StrictBool | None] = Field(
        default_factory=dict
    )
    reasonCode: str = Field(min_length=1, max_length=120)
    evidenceRefs: list[str] = Field(default_factory=list)
    selectedRemedy: str | None = Field(None, min_length=1, max_length=120)
    rulingEvidenceRef: str | None = Field(None, min_length=1)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_ruling_evidence(self):
        if (self.selectedRemedy is None) != (self.rulingEvidenceRef is None):
            raise ValueError(
                "selectedRemedy and rulingEvidenceRef must be supplied together"
            )
        if len(set(self.evidenceRefs)) != len(self.evidenceRefs):
            raise ValueError("adjudication evidenceRefs must be unique")
        AuthoredAdjudicationEventV1.build(self.model_dump(exclude_none=True))
        return self


class ExactTracePrefixV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1] = 1
    deckOrder: list[int] = Field(min_length=32, max_length=32)
    dealer: Literal["north", "west", "south", "east"]
    roundNumber: int = Field(1, ge=1)
    matchScore: list[int] = Field(
        default_factory=lambda: [0, 0], min_length=2, max_length=2
    )
    auctionActions: list[ExactAuctionActionV1Request] = Field(default_factory=list)
    escalationActions: list[ExactEscalationActionV1Request] = Field(
        default_factory=list
    )
    projectActions: list[ExactProjectActionV1Request] | None = None
    projectDeclarations: list[ExactProjectDeclarationV1Request] = Field(
        default_factory=list
    )
    projectAssertions: list[ExactProjectAssertionV1Request] = Field(
        default_factory=list
    )
    playActions: list[ExactPlayActionV1Request] = Field(default_factory=list)
    adjudicationEvents: list[ExactAdjudicationEventV1Request] = Field(
        default_factory=list
    )
    stateAssertions: dict[str, Any] = Field(default_factory=dict)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_deck(self):
        if sorted(self.deckOrder) != list(range(32)):
            raise ValueError("deckOrder must be a permutation of 0..31")
        if any(type(value) is not int or value < 0 for value in self.matchScore):
            raise ValueError("matchScore must contain two non-negative integers")
        if "sha256" in self.model_fields_set and self.sha256 is None:
            raise ValueError(
                "exact trace prefix sha256 must be a lowercase SHA-256 string"
            )
        return self


class ExactTraceLegalActionsPreviewV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1] = 1
    configuration: ResearchConfigurationSelectorRequest
    exactTracePrefix: ExactTracePrefixV1Request
    recordedEscalationOpportunities: list["RecordedEscalationOpportunityV1Request"] | None = None

    @field_validator("schemaVersion", mode="before")
    @classmethod
    def validate_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("exact trace preview schemaVersion must be integer 1")
        return value


class RecordedEscalationOpportunityV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    actor: Literal["north", "west", "south", "east"]
    actionCode: str = Field(min_length=1)


class ScenarioOriginV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    kind: Literal["search", "exact_trace_prefix"]
    exactTracePrefix: ExactTracePrefixV1Request | None = None
    recordedEscalationOpportunities: list[RecordedEscalationOpportunityV1Request] | None = None
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_origin(self):
        if self.kind == "exact_trace_prefix" and self.exactTracePrefix is None:
            raise ValueError("exact_trace_prefix origin requires exactTracePrefix")
        if self.kind == "search" and self.exactTracePrefix is not None:
            raise ValueError("search origin cannot include exactTracePrefix")
        if "recordedEscalationOpportunities" in self.model_fields_set:
            if self.recordedEscalationOpportunities is None:
                raise ValueError("recorded escalation opportunities cannot be null")
            ScenarioOriginV1.build(self.model_dump(exclude_none=True, exclude_unset=True))
        return self


class ScenarioDecisionPointV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    kind: Literal["bid", "escalation", "project", "play"]
    selector: dict[str, Any] = Field(default_factory=dict)


class ResearchScenarioV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1] = 1
    origin: ScenarioOriginV1Request
    constraints: ScenarioExpressionV1Request
    decisionPoint: ScenarioDecisionPointV1Request
    outcome: ScenarioExpressionV1Request
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class ScenarioProofRequestV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1] = 1
    mode: Literal["exact", "bounded", "policy_conditioned"]
    visibility: Literal["complete_state", "actor_information_set"]
    strategyClass: Literal["pure_deterministic", "frozen_policy"] | None = None
    claim: Literal[
        "outcome_reachable",
        "outcome_forced",
        "winning_root_action_count",
        "winning_continuation_line_count",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
    ]
    targetTeam: Literal["north_south", "west_east"]
    utility: str = Field(min_length=1, max_length=100)
    successThreshold: float = 0.0
    utilityEncoding: Literal["signed_win_loss_v1"] | None = None
    horizon: dict[str, Any] | None = None
    boundaryEvaluator: str | None = None
    lineUniverse: (
        Literal["all_legal", "optimal_strategy", "policy_conditioned"] | None
    ) = None
    policies: dict[str, Any] | None = None
    selectedAction: str | None = Field(None, min_length=1, max_length=32)
    budgetProfile: str = Field(min_length=1, max_length=120)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_proof_mode(self):
        if self.utilityEncoding is not None and self.utility != "contract_success":
            raise ValueError("utilityEncoding is valid only for contract_success")
        expected_strategy = (
            "frozen_policy"
            if self.mode == "policy_conditioned"
            else "pure_deterministic"
        )
        if self.visibility == "actor_information_set" and self.strategyClass is None:
            raise ValueError("actor-information-set proof requires strategyClass")
        if self.strategyClass is not None and self.strategyClass != expected_strategy:
            raise ValueError("strategyClass is incompatible with proof mode")
        if self.mode == "bounded":
            if self.horizon is None or not self.boundaryEvaluator:
                raise ValueError("bounded proof requires horizon and boundaryEvaluator")
        elif self.horizon is not None or self.boundaryEvaluator is not None:
            raise ValueError("horizon is valid only for bounded proof")
        if self.mode == "policy_conditioned":
            if self.policies is None or set(self.policies) != {
                "north",
                "west",
                "south",
                "east",
            }:
                raise ValueError(
                    "policy-conditioned proof requires all four seat policies"
                )
        elif self.policies is not None:
            raise ValueError("policies are valid only for policy-conditioned proof")
        if (
            self.claim == "winning_continuation_line_count"
            and self.lineUniverse is None
        ):
            raise ValueError("continuation-line proof requires lineUniverse")
        if (
            self.claim == "selected_action_is_unique_winner"
            and self.selectedAction is None
        ):
            raise ValueError("selected_action_is_unique_winner requires selectedAction")
        return self


class ResearchResourcesV4Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    generationLane: Literal["ortools"] = "ortools"
    generationPolicy: Literal["ortools_cp_sat_v3"] = "ortools_cp_sat_v3"
    seedStart: int = Field(1, ge=0)
    seedCount: int = Field(4, ge=1)
    seedsPerShard: int = Field(4, ge=1)
    shardWorkerCount: int | None = Field(default=None, ge=1)
    solverTimeoutSeconds: float = Field(15.0, gt=0.0)
    proofBudgetProfile: str = Field("late_game_exact_v1", min_length=1)

    @model_validator(mode="after")
    def validate_shards(self):
        if self.seedCount % self.seedsPerShard != 0:
            raise ValueError("seedCount must be divisible by seedsPerShard")
        return self


class ResearchResourcesLegacyV4Response(ResearchResourcesV3Request):
    """Read-only resource contract for immutable pre-OR-Tools-only V4 recipes."""

    generationPolicy: Literal[
        "natural_policy",
        "scenario_guided_actor_safe",
        "ortools_cp_sat_v2",
        "ortools_cp_sat_v3",
    ]
    proofBudgetProfile: str = Field(min_length=1)


class ResearchDiversityV4Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    cellDimensions: list[str] = Field(
        default_factory=lambda: [
            "contract",
            "project",
            "decision_kind",
            "proof_mode",
        ],
        min_length=1,
    )
    topKPerCell: int = Field(3, ge=1)
    rejectedAuditModulus: int = Field(100, ge=1)


class PlayingQualityAnalysisV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    utility: Literal[*PROOF_UTILITIES]
    utilityEncoding: Literal["signed_win_loss_v1"] | None = None
    budgetProfile: str = Field(min_length=1)
    horizon: dict[str, Any] | None = None
    boundaryEvaluator: str | None = None


class PlayingQualityPolicyV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1, 2]
    visibility: Literal["actor_information_set", "complete_state"] | None = None
    mode: Literal[*PLAYING_QUALITY_MODES]
    decisions: dict[Literal[*SCENARIO_DECISION_KINDS], PlayingQualityAnalysisV1Request]
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def validate_policy(cls, value):
        try:
            return ScenarioPlayingQualityPolicyV1.build(value).to_dict()
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


class ResearchAuctionSeedV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    seed: int = Field(ge=0)
    dealer: Literal["north", "west", "south", "east"]


class ResearchGenerationV1Request(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1]
    catalogHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tagQuery: dict[str, list[str]] = Field(json_schema_extra=tag_query_schema_v1())
    auctionSeeds: list[ResearchAuctionSeedV1Request] | SkipJsonSchema[None] = Field(
        None, min_length=1, json_schema_extra={"uniqueItems": True})
    sha256: str | SkipJsonSchema[None] = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def validate_generation(cls, value):
        return build_family_generation_request_v1(value)

    @model_serializer(mode="wrap")
    def omit_absent_auction_seeds(self, handler):
        value = handler(self)
        if self.auctionSeeds is None:
            value.pop("auctionSeeds", None)
        return value


class ResearchRecipeV4CreateRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[4]
    code: str | None = Field(None, pattern=r"^RCP-[A-Z0-9]{6,20}$")
    titleAr: str = Field(min_length=3, max_length=160)
    output: ResearchOutputV3Request = Field(default_factory=ResearchOutputV3Request)
    configuration: ResearchConfigurationSelectorRequest
    scenario: ResearchScenarioV1Request
    proof: ScenarioProofRequestV1Request
    playingQuality: PlayingQualityPolicyV1Request | None = None
    generation: ResearchGenerationV1Request | SkipJsonSchema[None] = None
    qualification: ResearchQualificationRequest = Field(
        default_factory=ResearchQualificationRequest
    )
    resources: ResearchResourcesV4Request = Field(
        default_factory=ResearchResourcesV4Request
    )
    diversity: ResearchDiversityV4Request = Field(
        default_factory=ResearchDiversityV4Request
    )
    renderPolicy: ResearchRenderPolicyV4Request = Field(
        default_factory=ResearchRenderPolicyV4Request
    )
    status: Literal["draft", "published", "retired"] = "published"

    @model_validator(mode="after")
    def validate_origin_policy(self):
        if "playingQuality" in self.model_fields_set and self.playingQuality is None:
            raise ValueError("playingQuality must be an object when supplied")
        if "generation" in self.model_fields_set and self.generation is None:
            raise ValueError("generation must be an object when supplied")
        if self.output.kind == "laffa" and self.output.resultCount != 1:
            raise ValueError("single laffa output requires exactly one result")
        if (
            self.scenario.origin.kind == "exact_trace_prefix"
            and self.output.resultCount != 1
        ):
            raise ValueError("exact trace prefix can produce exactly one result")
        if self.renderPolicy.videoCount > self.output.resultCount:
            raise ValueError("renderPolicy.videoCount cannot exceed output.resultCount")
        if self.renderPolicy.mode == "disabled" and self.renderPolicy.videoCount != 0:
            raise ValueError("disabled rendering requires videoCount zero")
        return self


class ResearchRecipeV4ConversionPreviewRequest(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    decisionPoint: ScenarioDecisionPointV1Request | None = None
    outcome: ScenarioExpressionV1Request | None = None
    proof: ScenarioProofRequestV1Request | None = None


class ResearchRunCreateRequest(BaseModel):
    model_config = {"extra": "forbid"}
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")


class ComparisonCampaignCreateRequest(BaseModel):
    model_config = {"extra": "forbid"}
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")


class ResearchQualificationResponse(BaseModel):
    """Legacy-readable qualification evidence; V3 requests stay fail-closed."""

    model_config = {"extra": "forbid"}
    minimumTruth: Literal[
        "exact_optimal",
        "robust_best_known",
        "debatable",
        "provisional",
        "rejected",
    ] = "robust_best_known"
    minimumConfidence: float = Field(0.75, ge=0.0, le=1.0)
    minimumSurroundingQuality: float = Field(70.0, ge=0.0, le=100.0)
    requireObserverSafety: bool = True
    requireAllDecisionAudit: bool = True
    maximumRegret: float = Field(0.0, ge=0.0)


class ResearchRecipeResponse(ResearchRecipeRequest):
    qualification: ResearchQualificationResponse = Field(
        default_factory=ResearchQualificationResponse
    )
    code: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycleStatus: Literal["draft", "published", "retired"] | None = None


class ResearchRecipeV2Response(ResearchRecipeV2Request):
    qualification: ResearchQualificationResponse = Field(
        default_factory=ResearchQualificationResponse
    )
    code: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycleStatus: Literal["draft", "published", "retired"] | None = None


class ResearchRecipeV3Response(ResearchRecipeV2Response):
    schemaVersion: Literal[3] = 3
    # Optional only for immutable V4 identities authored before output became
    # canonical. New V4 authoring always materializes this field.
    output: ResearchOutputV3Request | None = None
    situation: ResearchSituationV3Request
    tactics: list[ResearchTacticV3Request] = Field(default_factory=list)
    qualification: ResearchQualificationRequest = Field(
        default_factory=ResearchQualificationRequest
    )
    resources: ResearchResourcesV3Request = Field(
        default_factory=ResearchResourcesV3Request
    )
    diversity: ResearchDiversityV3Request = Field(
        default_factory=ResearchDiversityV3Request
    )
    configuration: ResearchConfigurationAuthorityResponse


class ScenarioDiagnosticV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    issueId: str
    class_: Literal[
        "schema_invalid",
        "rules_inapplicable",
        "unsupported_predicate",
        "contradictory_constraints",
        "structurally_infeasible",
        "not_found_within_budget",
        "normalization_incomplete",
        "construction_target_miss",
        "semantic_postcondition_miss",
        "proof_incomplete",
        "proof_failed",
        "satisfied",
    ] = Field(alias="class")
    severity: Literal["error", "warning", "info"]
    code: str
    predicateIds: list[str]
    fieldPaths: list[str]
    messageArKey: str
    messageEnKey: str
    details: dict[str, Any]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioCapabilityPlanRowV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    predicateId: str
    predicateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str
    hardness: Literal["hard", "prefer", "report"]
    constructionCapability: Literal[
        "cp_sat_exact",
        "incremental_actor_safe",
        "guided_search",
        "postcondition_only",
        "exact_input_only",
        "unsupported",
    ]
    runtimeVerifier: str
    proofModes: list[Literal["exact", "bounded", "policy_conditioned"]]
    hiddenInformation: Literal[
        "public",
        "actor_visible",
        "complete_state_only",
        "information_set_safe",
    ]
    costClass: Literal[
        "constant",
        "linear",
        "solver",
        "tree_search",
        "hidden_world_search",
    ]
    missingDomainPrimitives: list[str]


class ScenarioCapabilityPlanV1Response(BaseModel):
    """Legacy ScenarioCompilerV1 capability plan, not native V3 admission evidence."""

    model_config = {"extra": "forbid", "populate_by_name": True}
    schemaVersion: Literal[1] = 1
    scenarioHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proofHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: list[ScenarioCapabilityPlanRowV1Response]
    diagnostics: list[ScenarioDiagnosticV1Response]
    valid: bool = Field(
        title="Legacy compiler validity",
        description=(
            "Reports frozen ScenarioCompilerV1 support, not native V3 admission. "
            "It may be false for an admissible native V3 recipe; use "
            "nativeCapabilityPlan from native validation."
        ),
    )
    constructionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtimeVerificationHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proofPlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NormalizedExactStateV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    prefixHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: str
    decisionKind: Literal["bid", "escalation", "project", "play"]
    nextActor: Literal["north", "west", "south", "east"] | None = None
    legalActions: list[dict[str, Any]]
    state: dict[str, Any]
    projectDeclarationsVerified: int = Field(ge=0)
    projectAssertionsVerified: int = Field(ge=0)
    eventChainHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRecipeV4Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[4] = 4
    code: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    titleAr: str = Field(min_length=3, max_length=160)
    output: ResearchOutputV3Request = Field(default_factory=ResearchOutputV3Request)
    configuration: ResearchConfigurationAuthorityResponse
    scenario: ResearchScenarioV1Request
    proof: ScenarioProofRequestV1Request
    playingQuality: PlayingQualityPolicyV1Request | None = None
    generation: ResearchGenerationV1Request | SkipJsonSchema[None] = None
    qualification: ResearchQualificationRequest
    resources: ResearchResourcesV4Request | ResearchResourcesLegacyV4Response
    diversity: ResearchDiversityV4Request
    renderPolicy: ResearchRenderPolicyV4Request
    status: Literal["draft", "published", "retired"]
    capabilityPlan: ScenarioCapabilityPlanV1Response
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycleStatus: Literal["draft", "published", "retired"] | None = None

    @model_validator(mode="after")
    def validate_playing_quality(self):
        if "playingQuality" in self.model_fields_set and self.playingQuality is None:
            raise ValueError("playingQuality must be an object when supplied")
        if "generation" in self.model_fields_set and self.generation is None:
            raise ValueError("generation must be an object when supplied")
        return self


class ResearchRecipeV4ConversionSourceResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[3] = 3
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    recipeHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRecipeV4ConversionMappedFieldResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    sourcePath: str
    targetPaths: list[str]
    mappingKind: str


class ResearchRecipeV4ConversionUnmappedFieldResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    fieldPath: str
    reasonCode: str
    requiresOperatorReview: bool
    sourceValue: Any


class ResearchRecipeV4ConversionDiagnosticResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    code: str
    fieldPath: str
    message: str
    details: dict[str, Any] | None = None


class ResearchRecipeV4ConversionPreviewResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    source: ResearchRecipeV4ConversionSourceResponse
    proposedRecipe: dict[str, Any]
    targetRecipeCode: str | None = Field(None, pattern=r"^RCP-[A-Z0-9]{6,20}$")
    targetRecipeHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    mappedFields: list[ResearchRecipeV4ConversionMappedFieldResponse]
    unmappedFields: list[ResearchRecipeV4ConversionUnmappedFieldResponse]
    requiredOperatorFields: list[
        Literal["proof", "scenario.decisionPoint", "scenario.outcome"]
    ]
    buildDiagnostics: list[ResearchRecipeV4ConversionDiagnosticResponse]
    exactlyRepresentable: bool
    buildable: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


ResearchRecipeWireRequest = Annotated[
    ResearchRecipeRequest
    | ResearchRecipeV2Request
    | ResearchRecipeV3CreateRequest
    | ResearchRecipeV4CreateRequest,
    Field(discriminator="schemaVersion"),
]
ResearchRecipeCreateWireRequest = ResearchRecipeV4CreateRequest
ResearchRecipeWireResponse = Annotated[
    ResearchRecipeResponse
    | ResearchRecipeV2Response
    | ResearchRecipeV3Response
    | ResearchRecipeV4Response,
    Field(discriminator="schemaVersion"),
]


class StoredResearchRecipeResponse(ResearchRecipeResponse):
    lifecycleStatus: Literal["draft", "published", "retired"]


class StoredResearchRecipeV2Response(ResearchRecipeV2Response):
    lifecycleStatus: Literal["draft", "published", "retired"]


class StoredResearchRecipeV3Response(ResearchRecipeV3Response):
    lifecycleStatus: Literal["draft", "published", "retired"]


class StoredResearchRecipeV4Response(ResearchRecipeV4Response):
    lifecycleStatus: Literal["draft", "published", "retired"]
    nativeCapabilityPlan: NativeCapabilityPlanV3Response | None = None
    # Historical V4 recipes used the V3 resource shape. Keep the read contract
    # compatible without reopening those fields for new V4 authoring.
    resources: ResearchResourcesV4Request | ResearchResourcesLegacyV4Response


StoredResearchRecipeWireResponse = Annotated[
    StoredResearchRecipeResponse
    | StoredResearchRecipeV2Response
    | StoredResearchRecipeV3Response
    | StoredResearchRecipeV4Response,
    Field(discriminator="schemaVersion"),
]


class ResearchValidationIssueResponse(BaseModel):
    model_config = {"extra": "forbid"}

    severity: Literal["error", "warning", "info"]
    code: str
    field: str | None = None
    messageAr: str | None = None
    messageEn: str | None = None
    primitiveIds: list[str] = Field(default_factory=list)
    issueId: str | None = None
    class_: str | None = Field(None, alias="class")
    predicateIds: list[str] = Field(default_factory=list)
    fieldPaths: list[str] = Field(default_factory=list)
    messageArKey: str | None = None
    messageEnKey: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class NativeCapabilityPlanRowV3Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    predicateId: str
    predicateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str
    hardness: Literal["hard", "prefer", "report"]
    nativeStatus: Literal["native_exact", "unsupported"]
    capabilityHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    qualificationBinding: Literal["cp_sat_literal", "proof_variable"] | None = None


class NativeCapabilityPlanV3Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[3]
    backendIdentityHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalogHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenarioHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proofHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: list[NativeCapabilityPlanRowV3Response]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRecipeValidationResponse(BaseModel):
    model_config = {"extra": "forbid", "populate_by_name": True}

    schemaVersion: Literal[1, 2] = 1
    valid: bool
    recipe: ResearchRecipeWireResponse
    issues: list[ResearchValidationIssueResponse]
    scenarioPlanHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    configurationAuthority: ResearchConfigurationAuthorityResponse | None = None
    capabilityPlan: ScenarioCapabilityPlanV1Response | None = None
    nativeCapabilityPlan: NativeCapabilityPlanV3Response | None = None
    normalizedExactState: NormalizedExactStateV1Response | None = None


class ResearchEstimateOutputV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    kind: Literal["laffa", "laffa_collection"]
    requestedResultCount: int = Field(ge=1, le=1000)
    requestedVideoCount: int = Field(ge=0, le=100)
    maximumCandidateCount: int = Field(ge=1)
    maximumVariantCount: int = Field(ge=1)
    admissible: bool
    reason: Literal[
        "requested_seed_count_exceeds_candidate_budget",
        "requested_result_count_exceeds_variant_budget",
        "requested_result_count_exceeds_candidate_budget",
    ] | None = None
    recommendedResultCount: int = Field(ge=1, le=1000)
    requiresResourceExpansion: bool


class ResearchEstimateBackendIdentityV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    backendId: Literal["ortools_cp_sat_v2", "ortools_cp_sat_v3"]
    backendVersion: str
    qualificationAuthority: Literal["cp_sat"]
    solverVersion: str
    planner: Literal["tail_first_v1"]
    deterministicExecution: Literal[True]
    bundleRevisionId: str
    bundleIdentityHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorityHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicateCatalogHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    loweringManifestHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchEstimateConstructionPlanV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    planner: Literal["generic", "tail_first"]
    fallbackReason: str | None = None
    prefixLength: int | None = Field(None, ge=0, le=31)
    tailOffset: int | None = Field(None, ge=0, le=31)
    tailLength: int | None = Field(None, ge=1, le=32)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchEstimateShardingV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    configuredSeedsPerShard: int = Field(ge=1)
    requestedShardCount: int = Field(ge=1)
    actualShardCount: int = Field(ge=1)
    maximumShardWorkers: int = Field(ge=1)
    requestedShardWorkers: int | None = Field(default=None, ge=1)
    effectiveShardWorkers: int = Field(ge=1)
    maximumShardWallSeconds: float | None = Field(default=None, gt=0.0)
    parallelismLimitedByShardPlan: bool
    recommendedSeedsPerShard: int = Field(ge=1)


class ResearchEstimateHorizonV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    unit: Literal["plies", "decisions", "tricks", "round_end"]
    value: int = Field(ge=1)


class ResearchEstimateProofV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    requestedMode: Literal["exact", "bounded", "policy_conditioned"]
    visibility: Literal["complete_state", "actor_information_set"]
    strategyClass: Literal["pure_deterministic", "frozen_policy"]
    horizon: ResearchEstimateHorizonV2Response | None = None
    admissible: bool
    reason: str | None = None
    recommendedMode: Literal["bounded", "policy_conditioned"] | None = None
    recommendedModes: list[Literal["bounded", "policy_conditioned"]]
    estimatedRootActionUpperBound: int = Field(ge=0)
    actionDomainUpperBound: int = Field(ge=1)
    estimatedEkkaActionVariantUpperBound: int = Field(ge=0, le=3)
    ekkaDeclarationSequenceUpperBound: str
    targetInformationSetUpperBound: int | None = Field(None, ge=0)
    opponentInformationSetUpperBound: int | None = Field(None, ge=0)
    targetPolicyCountUpperBound: str | None = None
    opponentPolicyCountUpperBound: str | None = None
    policyProductUpperBound: str | None = None
    estimatedTreeStateUpperBound: int | None = Field(None, ge=0)
    policyProductExceedsBudget: bool | None = None
    informationTreeExceedsBudget: bool | None = None
    estimatedStateSpaceClass: str
    hiddenWorldClass: str
    budgetProfile: str
    resourceBudgetProfile: str
    remainingCardCount: int | None = Field(None, ge=0)
    requestedDepth: int | None = Field(None, ge=0)
    budgetMaximumDepth: int | None = Field(None, ge=1)
    budgetMaximumHorizonActions: int | None = Field(None, ge=1)
    estimatedStateSpace: int | None = Field(None, ge=1)
    estimatedHiddenWorldUpperBound: int | None = Field(None, ge=1)
    budgetMaximumHiddenWorlds: int | None = Field(None, ge=1)
    budgetStateSpaceLimit: int | None = Field(None, ge=1)
    budgetProfileHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    proofBudgetRegistryHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRecipeEstimateResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}
    schemaVersion: Literal[1, 2]
    seedCount: int = Field(ge=1)
    candidateCount: int = Field(ge=1)
    shardCount: int = Field(ge=1)
    sharding: ResearchEstimateShardingV2Response | None = None
    maximumSolverSeconds: float = Field(ge=0.0)
    maximumCampaignSeconds: float | None = Field(None, ge=0.0)
    analysisDecisionUpperBound: int = Field(ge=0)
    warnings: list[str]
    tacticalVerificationDecisionUpperBound: int = Field(ge=0)
    valid: bool
    issues: list[ResearchValidationIssueResponse]
    scenarioPlanHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    capabilityPlanHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    nativeCapabilityPlanHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    predicatesByConstructionCapability: dict[str, list[str]] | None = None
    postconditionOnlyPredicates: list[str] | None = None
    exactInputOnlyPredicates: list[str] | None = None
    unsupportedPredicates: list[str] | None = None
    output: ResearchEstimateOutputV2Response | None = None
    proof: ResearchEstimateProofV2Response | None = None
    expectedCostCategory: Literal[
        "offline_heavyweight",
        "bounded",
        "late_game_exact",
        "policy_conditioned",
        "not_compiled",
    ] | None = None
    launchAdmissible: bool | None = None
    configurationAuthority: ResearchConfigurationAuthorityResponse | None = None
    backendIdentity: ResearchEstimateBackendIdentityV2Response | None = None
    compiledStages: list[str] | None = None
    constructionPlan: ResearchEstimateConstructionPlanV1Response | None = None
    resourceTier: str | None = None
    refusalReason: str | None = None
    estimatedModelVariables: int | None = Field(None, ge=0)
    estimatedModelConstraints: int | None = Field(None, ge=0)
    estimatedGraphNodes: int | None = Field(None, ge=1)
    estimatedGraphEdges: int | None = Field(None, ge=0)
    estimatedHiddenWorlds: int | str | None = None
    estimatedCountLimbs: int | None = Field(None, ge=0)
    firstResultCostClass: Literal[
        "within_selected_tier", "exceeds_selected_tier", "not_estimable"
    ] | None = None
    requestedResultsCostClass: Literal[
        "within_selected_tier", "exceeds_selected_tier", "not_estimable"
    ] | None = None
    alternativeProofModes: list[
        Literal["exact", "bounded", "policy_conditioned"]
    ] | None = None
    requestedCombinationSupported: bool | None = None
    exhaustiveSearchInfeasibilityEstablishable: bool | None = None
    estimateHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_v2_campaign_limit(cls, value):
        if (
            isinstance(value, Mapping)
            and value.get("schemaVersion") == 2
            and "maximumCampaignSeconds" not in value
            and "backendIdentity" in value
        ):
            return {
                **value,
                "maximumCampaignSeconds": value.get("maximumSolverSeconds"),
            }
        return value

    @field_validator("predicatesByConstructionCapability")
    @classmethod
    def validate_capability_buckets(cls, value):
        if value is None:
            return value
        supported = {
            "cp_sat_exact",
            "incremental_actor_safe",
            "guided_search",
            "postcondition_only",
            "exact_input_only",
            "unsupported",
        }
        if set(value) - supported:
            raise ValueError("estimate construction capability is unsupported")
        if any(items != sorted(set(items)) for items in value.values()):
            raise ValueError("estimate predicate capability buckets must be canonical")
        return value

    @field_validator("estimatedHiddenWorlds", mode="before")
    @classmethod
    def validate_hidden_world_estimate(cls, value):
        if value is None or (type(value) is int and value >= 1):
            return value
        if type(value) is str and re.fullmatch(r"[1-9][0-9]*", value):
            return value
        raise ValueError("estimatedHiddenWorlds must be a positive integer or canonical decimal string")

    @model_validator(mode="after")
    def validate_versioned_shape(self):
        historical_v2_fields = (
            "scenarioPlanHash",
            "capabilityPlanHash",
            "predicatesByConstructionCapability",
            "postconditionOnlyPredicates",
            "exactInputOnlyPredicates",
            "unsupportedPredicates",
            "output",
            "proof",
            "expectedCostCategory",
            "launchAdmissible",
            "configurationAuthority",
            "estimateHash",
        )
        cp_sat_v2_fields = (
            "maximumCampaignSeconds",
            "backendIdentity",
            "compiledStages",
            "resourceTier",
            "refusalReason",
            "estimatedModelVariables",
            "estimatedModelConstraints",
            "estimatedGraphNodes",
            "estimatedGraphEdges",
            "estimatedHiddenWorlds",
            "estimatedCountLimbs",
            "firstResultCostClass",
            "requestedResultsCostClass",
            "alternativeProofModes",
            "requestedCombinationSupported",
            "exhaustiveSearchInfeasibilityEstablishable",
        )
        historical_present = set(historical_v2_fields) & self.model_fields_set
        cp_sat_present = set(cp_sat_v2_fields) & self.model_fields_set
        if self.schemaVersion == 2:
            if len(historical_present) != len(historical_v2_fields) or (
                cp_sat_present and len(cp_sat_present) != len(cp_sat_v2_fields)
            ):
                raise ValueError("Research Estimate V2 fields are incomplete")
        elif historical_present or cp_sat_present:
            raise ValueError("Research Estimate V1 cannot contain V2 fields")
        if self.nativeCapabilityPlanHash is not None and (
            self.schemaVersion != 2
            or self.backendIdentity is None
            or self.backendIdentity.backendId != "ortools_cp_sat_v3"
        ):
            raise ValueError("nativeCapabilityPlanHash requires the native V3 backend")
        return self


class ResearchLaunchAdmissionDiagnosticResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    code: Literal["research_launch_not_admissible"]
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    recipeHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configurationAuthority: ResearchConfigurationAuthorityResponse
    scenarioPlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str
    recommendedModes: list[Literal["bounded", "policy_conditioned"]]
    output: ResearchEstimateOutputV2Response
    proof: ResearchEstimateProofV2Response
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchLaunchAdmissionErrorResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    detail: ResearchLaunchAdmissionDiagnosticResponse


class ResearchRuleCoverageEvidenceResponse(BaseModel):
    model_config = {"extra": "forbid"}

    runtime: list[str]
    solver: list[str]
    documentation: list[str]
    tests: list[str]
    proofBoundary: str


class ResearchRulePrimitiveResponse(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    kind: str
    enabled: bool
    runtimeCoverage: bool
    solverCoverage: bool
    compilerStage: str
    cpSatNative: bool
    documentationCoverage: bool
    testCoverage: bool
    coverageEvidence: ResearchRuleCoverageEvidenceResponse
    payloadHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchPluginContractResponse(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    version: str
    safetyGuards: list[str]
    cancellationMode: str
    failureMode: str
    rollbackSupported: bool


class ResearchPluginDescriptorResponse(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    kind: str
    version: str
    apiVersion: str
    description: str
    capabilities: list[str]
    resourceClass: str
    deterministic: bool
    hostApiRange: str
    profileSchemaRange: str
    isolation: str
    permissions: list[str]
    sideEffects: list[str]
    artifactSha256: str | None
    signed: bool
    timeoutSeconds: int = Field(ge=1)
    maximumMemoryMb: int = Field(ge=1)
    trustClass: str
    compatibleProfileSchemas: list[str]
    signature: str | None
    contract: ResearchPluginContractResponse | None = None
    external: bool | None = None
    source: str | None = None


class ResearchDomainPrimitiveLoweringsResponse(BaseModel):
    model_config = {"extra": "forbid"}

    runtime: str | None
    reference: str | None
    cpSat: str | None
    policy: str | None


class ResearchDomainPrimitiveDefinitionResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    kind: str
    version: str
    category: str
    requiredTargets: list[str]
    lowerings: ResearchDomainPrimitiveLoweringsResponse
    documentation: list[str]
    tests: list[str]
    influenceProof: str | None


class ResearchDomainPrimitiveInstanceResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    primitiveId: str
    definitionKind: str
    category: str
    valueHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requiredTargets: list[str]
    coveredTargets: list[str]
    missingTargets: list[str]
    coverageStatus: str
    executableBindingStatus: str
    behavioralOrInvariantStatus: str
    declaredBehavioralEvidence: str | None
    verifiedBehavioralEvidence: str | None
    qualificationEligible: bool
    value: Any = None


class ResearchDomainCompilerCoverageResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    covered: int = Field(ge=0)
    total: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)
    complete: bool
    executableBindingCovered: int = Field(ge=0)
    executableBindingTotal: int = Field(ge=0)
    executableBindingPercentage: float = Field(ge=0.0, le=100.0)
    executableBindingComplete: bool
    behavioralOrInvariantCovered: int = Field(ge=0)
    behavioralOrInvariantTotal: int = Field(ge=0)
    behavioralOrInvariantPercentage: float = Field(ge=0.0, le=100.0)
    behavioralOrInvariantComplete: bool
    rows: list[ResearchDomainPrimitiveInstanceResponse]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchDomainIrResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    primitiveInstances: list[ResearchDomainPrimitiveInstanceResponse]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rules: dict[str, Any] | None = None
    conventions: dict[str, Any] | None = None
    localization: dict[str, Any] | None = None


class ResearchDomainCompilerTargetsResponse(BaseModel):
    model_config = {"extra": "forbid"}

    runtimePlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    referencePlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cpSatBlueprintHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policyPlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uiSchemaHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    documentationModelHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchCompiledDomainResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    compilerVersion: Literal["domain_compiler_v1"]
    domainIr: ResearchDomainIrResponse
    coverage: ResearchDomainCompilerCoverageResponse
    definitions: list[ResearchDomainPrimitiveDefinitionResponse]
    targets: ResearchDomainCompilerTargetsResponse
    publishable: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    targetPayloads: dict[str, Any] | None = None


class ResearchDomainCompilerV3Response(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[3]
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    conventionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    localizationHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiledDomain: ResearchCompiledDomainResponse
    pluginCapabilities: list[ResearchPluginDescriptorResponse]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchCapabilityShardingResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    deterministicSeedPartitions: bool
    durableLeases: bool
    fencingTokens: bool
    maximumLocalWorkers: int = Field(ge=1)
    leaseSeconds: float = Field(gt=0.0)
    heartbeatSeconds: float = Field(gt=0.0)
    maximumAttempts: int = Field(ge=1)
    reductionOrder: str


class ResearchCompilerQualificationResponse(BaseModel):
    model_config = {"extra": "forbid"}

    legacyResearchAvailable: bool
    requiresExecutableBindingCoverage: bool
    requiresBehavioralOrInvariantCoverage: bool
    executableBindingComplete: bool
    behavioralOrInvariantComplete: bool
    publicationEligible: bool


class ResearchOutputModesCapabilityResponse(BaseModel):
    model_config = {"extra": "forbid"}

    laffa: str
    laffa_collection: str
    contextualized_laffa: str
    sakka: str


class ResearchTacticalVerificationCapabilityResponse(BaseModel):
    model_config = {"extra": "forbid"}

    available: bool
    maximumDecisionsPerLaffa: int = Field(ge=0)
    supportedPatterns: list[str]
    defaultPolicy: str


class ResearchConfigurationSelectionReceiptResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    bundleRevisionId: str
    bundleId: str
    bundleVersion: str
    semanticBundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveRuntimeConfigHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contextHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    domainHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    domainIrHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    legacyDomainV2Hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ready: Literal[True]


class ScenarioPredicateDefinitionV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    kind: str
    version: Literal[1]
    category: str
    stage: Literal[
        "deal",
        "auction",
        "escalation",
        "projects",
        "playPrefix",
        "state",
        "outcome",
        "tactics",
        "proof",
    ]
    argumentSchema: dict[str, Any]
    messageArKey: str
    messageEnKey: str
    requiredDomainPrimitives: list[str]
    allowedHardness: list[Literal["hard", "prefer", "report"]]
    constructionCapability: Literal[
        "cp_sat_exact",
        "incremental_actor_safe",
        "guided_search",
        "postcondition_only",
        "exact_input_only",
        "unsupported",
    ]
    runtimeVerifier: str
    supportedProofModes: list[Literal["exact", "bounded", "policy_conditioned"]]
    hiddenInformation: Literal[
        "public",
        "actor_visible",
        "complete_state_only",
        "information_set_safe",
    ]
    costClass: Literal[
        "constant",
        "linear",
        "solver",
        "tree_search",
        "hidden_world_search",
    ]
    safeComplement: bool
    examples: list[dict[str, Any]]
    counterexamples: list[dict[str, Any]]
    documentation: list[str]
    tests: list[str]
    lifecycleStatus: Literal["active", "deprecated", "retired"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioPredicateRegistryV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    registryVersion: Literal["scenario_predicate_registry_v1"]
    definitions: list[ScenarioPredicateDefinitionV1Response]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioCpSatStageSupportV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    status: Literal["exact", "derived", "unsupported"]
    stages: list[
        Literal[
            "deal",
            "auction",
            "escalation",
            "projects",
            "playPrefix",
            "state",
            "outcome",
            "tactics",
            "proof",
        ]
    ]


class ScenarioCpSatExactInputSupportV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    status: Literal["exact", "unsupported"]
    stages: list[
        Literal[
            "deal",
            "auction",
            "escalation",
            "projects",
            "playPrefix",
            "state",
            "outcome",
            "tactics",
            "proof",
        ]
    ]


class ScenarioCpSatProofSupportV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    exact: Literal["exact", "unsupported"]
    bounded: Literal["exact", "bounds", "unsupported"]
    policyConditioned: Literal["exact", "unsupported"]


class ScenarioCpSatSupportV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    search: ScenarioCpSatStageSupportV1Response
    exactInput: ScenarioCpSatExactInputSupportV1Response
    qualificationSource: Literal[
        "construction", "derived_state", "proof", "report_only"
    ]
    actorInformationSet: Literal[
        "public", "observation", "reference_world_only", "nonanticipative"
    ]
    proof: ScenarioCpSatProofSupportV1Response
    evidenceExtractor: str


class ScenarioPredicateDefinitionV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[2] = 2
    kind: str
    version: int = Field(ge=1)
    category: str
    stage: Literal[
        "deal",
        "auction",
        "escalation",
        "projects",
        "playPrefix",
        "state",
        "outcome",
        "tactics",
        "proof",
    ]
    argumentSchema: dict[str, Any]
    messageArKey: str
    messageEnKey: str
    requiredDomainPrimitives: list[str]
    allowedHardness: list[Literal["hard", "prefer", "report"]]
    cpSatSupport: ScenarioCpSatSupportV1Response
    costClass: Literal[
        "constant", "linear", "solver", "tree_search", "hidden_world_search"
    ]
    safeComplement: bool
    examples: list[dict[str, Any]]
    counterexamples: list[dict[str, Any]]
    documentation: list[str]
    tests: list[str]
    lifecycleStatus: Literal["active", "deprecated", "retired"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioPredicateCatalogV2Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[2] = 2
    backendIdentityHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["search", "exact_trace_prefix"]
    proofMode: Literal["exact", "bounded", "policy_conditioned"]
    visibility: Literal["complete_state", "actor_information_set"]
    hardness: Literal["hard", "prefer", "report"]
    definitions: list[ScenarioPredicateDefinitionV2Response]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OrToolsScenarioBackendIdentityV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    backendId: Literal["ortools_cp_sat_v2", "ortools_cp_sat_v3"]
    backendVersion: str
    qualificationAuthority: Literal["cp_sat"]
    solverVersion: str
    planner: Literal["tail_first_v1"]
    deterministicExecution: Literal[True]
    bundleRevisionId: str
    bundleIdentityHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorityHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicateCatalogHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioProofVisibilityCapabilityV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    mode: Literal["exact", "bounded", "policy_conditioned"]
    visibility: Literal["complete_state", "actor_information_set"]


class ScenarioExecutionCombinationV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    origin: Literal["search", "exact_trace_prefix"]
    decisionKind: Literal["bid", "escalation", "project", "play"]
    mode: Literal["exact", "bounded", "policy_conditioned"]
    visibility: Literal["complete_state", "actor_information_set"]
    claims: list[
        Literal[
            "outcome_reachable",
            "outcome_forced",
            "winning_root_action_count",
            "winning_continuation_line_count",
            "selected_action_is_unique_winner",
            "target_team_has_no_salvage",
        ]
    ] = Field(min_length=1)
    lineUniverses: list[
        Literal["all_legal", "optimal_strategy", "policy_conditioned"]
    ]
    strategyClasses: list[Literal["pure_deterministic", "frozen_policy"]] = Field(
        min_length=1,
        max_length=1,
    )
    contractValues: list[Literal["sun", "hokum", "ashkal"]] | None


class ScenarioExecutionCapabilitiesV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    combinations: list[ScenarioExecutionCombinationV1Response] = Field(min_length=1)
    phase4AcceptanceGaps: list[dict[str, JsonValue]] = Field(default_factory=list)
    decisionUnavailableReasons: dict[str, str] = Field(default_factory=dict)
    decisionSelectorSchemas: dict[str, dict[str, JsonValue]]
    authoringCatalog: dict[str, JsonValue]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("decisionSelectorSchemas")
    @classmethod
    def validate_decision_selector_schemas(cls, value):
        if set(value) - {"bid", "escalation", "project", "play"}:
            raise ValueError("scenario decision selector schema kind is unsupported")
        if "play" not in value:
            raise ValueError("scenario play selector schema is required")
        return value

    @field_validator("decisionUnavailableReasons")
    @classmethod
    def validate_decision_unavailable_reasons(cls, value):
        if set(value) - {"bid", "escalation", "project", "play"}:
            raise ValueError("scenario decision refusal kind is unsupported")
        if any(not reason.strip() for reason in value.values()):
            raise ValueError("scenario decision refusal reason is required")
        return value


class ScenarioResearchLocalizationGroupsV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    workflow: dict[str, str]

    @field_validator("workflow")
    @classmethod
    def validate_workflow_copy(cls, value):
        if not value or any(
            not key.strip() or not text.strip()
            for key, text in value.items()
        ):
            raise ValueError("scenario Research workflow copy is invalid")
        return value


class ScenarioResearchLocalizationV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    bundleRevisionId: str = Field(min_length=1, max_length=200)
    productRevisionId: str = Field(min_length=1, max_length=200)
    productProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    locale: Literal["ar-SA"]
    direction: Literal["rtl"]
    groups: ScenarioResearchLocalizationGroupsV1Response
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioResearchV4CapabilityV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    backendIdentity: OrToolsScenarioBackendIdentityV1Response
    executionReady: bool
    launchAdmissible: bool
    supportedOrigins: list[Literal["search", "exact_trace_prefix"]]
    supportedProofVisibility: list[ScenarioProofVisibilityCapabilityV1Response]
    executionCapabilities: ScenarioExecutionCapabilitiesV1Response
    hardPredicateCatalog: ScenarioPredicateCatalogV2Response
    localization: ScenarioResearchLocalizationV1Response
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_localization_bundle_authority(self):
        if self.localization.bundleRevisionId != self.backendIdentity.bundleRevisionId:
            raise ValueError("scenario Research localization bundle authority mismatch")
        origins = {item.origin for item in self.executionCapabilities.combinations}
        if origins != set(self.supportedOrigins):
            raise ValueError("scenario Research supported origins mismatch")
        proof_visibility = {
            (item.mode, item.visibility)
            for item in self.executionCapabilities.combinations
        }
        declared_proof_visibility = {
            (item.mode, item.visibility) for item in self.supportedProofVisibility
        }
        if proof_visibility != declared_proof_visibility:
            raise ValueError("scenario Research proof visibility mismatch")
        if self.executionReady != self.launchAdmissible:
            raise ValueError("scenario Research execution and launch capability disagree")
        return self


class NativeObservedProofLimitsV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    schemaVersion: Literal[1]
    maximumIterations: int = Field(ge=1, le=2**31 - 1)
    maximumQueries: int = Field(ge=1, le=2**31 - 1)
    maximumObservations: int = Field(ge=1, le=2**31 - 1)
    maximumBoundPrefixes: int = Field(ge=1, le=2**31 - 1)
    maximumCachedPrefixesPerBranch: int = Field(ge=1, le=2**31 - 1)

    @model_validator(mode="before")
    @classmethod
    def validate_native_limits(cls, value):
        from ..research.scenario_contracts import NativeObservedProofLimitsV1
        return NativeObservedProofLimitsV1.build(value).to_dict()


class NativeObservedProofLimitsV2Response(NativeObservedProofLimitsV1Response):
    schemaVersion: Literal[2]
    counterexampleObjective: Literal["minimum_utility"]


class NativeObservedProofLimitsV3Response(NativeObservedProofLimitsV2Response):
    schemaVersion: Literal[3]
    playAllocationMethod: Literal["deterministic_cp_sat_batched_prefix_v1"]


class ScenarioProofBudgetV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    profileId: str
    maximumNodes: int = Field(ge=1)
    maximumDepth: int = Field(ge=1)
    maximumHorizonActions: int = Field(ge=1)
    maximumHiddenWorlds: int = Field(ge=1)
    maximumWallSeconds: float = Field(gt=0.0)
    maximumCpuSeconds: float = Field(gt=0.0)
    maximumCacheEntries: int = Field(ge=1)
    maximumCacheBytes: int = Field(ge=1)
    cancellationCheckIntervalNodes: int = Field(ge=1)
    maximumWorkers: int = Field(ge=1)
    maximumLineCountNodes: int = Field(ge=1)
    maximumStateSpace: int = Field(ge=1)
    policySearch: Literal["time_budgeted_symbolic_v1"] | None = None
    nativeObserved: NativeObservedProofLimitsV1Response | NativeObservedProofLimitsV2Response | NativeObservedProofLimitsV3Response | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioProofBudgetRegistryV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profiles: dict[str, ScenarioProofBudgetV1Response]


class ScenarioPolicyProfileV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    agentId: Literal["heuristic_bot_v1"]
    version: Literal["1.0.0"]
    seed: int
    profileHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioDeclarativePolicyChoiceV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    nodeId: str
    actionId: int = Field(strict=True, ge=0)


class ScenarioDeclarativePolicyProfileV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    policyId: Literal["fixed_action_map_v1"]
    version: Literal["1.0.0"]
    seed: int = Field(strict=True, ge=0)
    choices: list[ScenarioDeclarativePolicyChoiceV1Response]
    materialization: Literal["seeded_legal_action_order_v1"]
    profileHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioProjectOptionV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    contract: Literal["sun", "hokum", "ashkal"]
    kind: str
    value: int = Field(ge=0)
    highRank: str
    highRankId: int = Field(ge=0, le=7)


class ScenarioCardOptionV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    id: int = Field(ge=0, le=31)
    code: str
    suit: Literal["spades", "hearts", "diamonds", "clubs"]
    rank: Literal[
        "seven",
        "eight",
        "nine",
        "jack",
        "queen",
        "king",
        "ten",
        "ace",
    ]
    symbol: str
    label: str


class ExactTracePreviewConfigurationAuthorityV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    bundleRevisionId: str = Field(min_length=1, max_length=200)
    semanticBundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveRuntimeConfigHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    backendIdentityHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExactTracePreviewProjectDeclarationV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    owner: Literal["north", "west", "south", "east"]
    kind: str = Field(min_length=1, max_length=80)
    value: int = Field(ge=0)
    highRank: int = Field(ge=0)
    declarationOrder: int = Field(ge=1)


class ExactTraceLegalActionV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    stage: Literal["auction", "escalation", "project", "play"]
    code: str = Field(min_length=1, max_length=100)
    actor: Literal["north", "west", "south", "east"] | None = None
    action: str | None = Field(None, min_length=1, max_length=40)
    trumpSuit: Literal["spades", "hearts", "diamonds", "clubs"] | None = None
    mode: Literal["open", "locked"] | None = None
    projectId: str | None = Field(None, pattern=r"^PROJECT-[0-9A-F]{16}$")
    actionId: int | None = Field(None, ge=0, le=63)
    card: ScenarioCardOptionV1Response | None = None
    ekka: bool | None = None
    projectDeclarations: list[ExactTracePreviewProjectDeclarationV1Response] = Field(
        default_factory=list,
        max_length=1,
    )

    @model_validator(mode="after")
    def validate_stage_shape(self):
        if self.actor is None:
            raise ValueError("exact trace legal action requires actor")
        if self.stage in {"auction", "escalation"} and self.action is None:
            raise ValueError("non-play exact trace action requires action")
        if self.stage == "project":
            if self.action not in {"pass", "declare"}:
                raise ValueError("project exact trace action is invalid")
            if self.action == "declare" and self.projectId is None:
                raise ValueError("project exact trace declare requires projectId")
            if self.action == "pass" and self.projectId is not None:
                raise ValueError("project exact trace pass cannot contain projectId")
        if self.stage == "play" and (
            self.actionId is None or self.card is None or self.ekka is None
        ):
            raise ValueError("play exact trace action requires card identity")
        if self.stage != "play" and (
            self.actionId is not None
            or self.card is not None
            or self.ekka is not None
            or self.projectDeclarations
        ):
            raise ValueError("non-play exact trace action has play-only fields")
        if self.stage != "project" and self.projectId is not None:
            raise ValueError("projectId is valid only for project actions")
        return self


class ExactTracePreviewDiagnosticV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    code: Literal["exact_trace_prefix_invalid", "exact_normalization_incomplete"]
    fieldPath: Literal["exactTracePrefix"]


class ExactTraceLegalActionsPreviewV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    configurationAuthority: ExactTracePreviewConfigurationAuthorityV1Response
    valid: bool
    currentStage: Literal["auction", "escalation", "project", "play", "terminal"] | None
    nextActor: Literal["north", "west", "south", "east"] | None
    terminal: bool
    legalActions: list[ExactTraceLegalActionV1Response]
    diagnostics: list[ExactTracePreviewDiagnosticV1Response]
    prefixHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    previewStateHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    previewReceiptHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_result_shape(self):
        identities = (
            self.prefixHash,
            self.previewStateHash,
            self.previewReceiptHash,
        )
        if self.valid:
            if self.currentStage is None or any(value is None for value in identities):
                raise ValueError("valid exact trace preview requires identities")
            if self.diagnostics:
                raise ValueError("valid exact trace preview cannot contain diagnostics")
            if self.terminal != (self.currentStage == "terminal"):
                raise ValueError("exact trace preview terminal status is inconsistent")
            if self.terminal and self.legalActions:
                raise ValueError("terminal exact trace preview cannot have legal actions")
        elif (
            self.currentStage is not None
            or any(value is not None for value in identities)
            or self.legalActions
            or not self.diagnostics
        ):
            raise ValueError("invalid exact trace preview must fail closed")
        return self


class ScenarioPlayActionOptionV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    actionId: int = Field(ge=0, le=63)
    code: str
    card: ScenarioCardOptionV1Response
    ekka: bool


class ScenarioAuctionActionsByRoundV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    round: Literal[1, 2]
    actions: list[Literal["pass", "sun", "hokum", "ashkal"]]


class ScenarioEscalationOptionsByContractV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    contract: Literal["sun", "hokum", "ashkal"]
    levels: list[Literal["normal", "double", "triple", "four", "gahwa"]]
    actions: list[Literal["decline", "double", "triple", "four", "gahwa"]]


class ScenarioRuleOptionsV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    contracts: list[Literal["sun", "hokum", "ashkal"]]
    auctionActionsByRound: list[ScenarioAuctionActionsByRoundV1Response]
    escalationLevelsByContract: list[ScenarioEscalationOptionsByContractV1Response]
    ashkalBuyerRelations: list[Literal["bidder", "bidder_partner"]]
    trumpContracts: list[Literal["sun", "hokum", "ashkal"]]
    balootContracts: list[Literal["sun", "hokum", "ashkal"]]
    playActionOptions: list[ScenarioPlayActionOptionV1Response]
    projectsEnabled: bool
    kaboutEnabled: bool


class ScenarioDecisionCapabilityV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    kind: Literal["bid", "escalation", "project", "play"]
    authoringSupported: bool
    exactTraceNormalizationSupported: bool
    searchGenerationSupported: bool
    proofModes: list[Literal["exact", "bounded", "policy_conditioned"]]
    renderPreparationSupported: bool
    unavailableReason: str | None


class ScenarioResultCountPolicyV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    minimum: Literal[1]
    maximum: Literal[1000]
    default: Literal[1]


class ScenarioVideoCountPolicyV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    minimum: Literal[0]
    maximum: Literal[100]
    default: Literal[1]
    maximumRelation: Literal["output.resultCount"]


class ScenarioOutputPolicyV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    kinds: list[Literal["laffa", "laffa_collection"]]
    resultCount: ScenarioResultCountPolicyV1Response
    renderModes: list[Literal["on_demand", "disabled"]]
    videoCount: ScenarioVideoCountPolicyV1Response


class ResearchScenarioConfigurationV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    recipeSchemaVersion: Literal[4]
    supportedLegacyReadVersions: list[Literal[1, 2, 3]]
    origins: list[Literal["search", "exact_trace_prefix"]]
    outputPolicy: ScenarioOutputPolicyV1Response
    decisionCapabilities: list[ScenarioDecisionCapabilityV1Response]
    sections: list[
        Literal[
            "deal",
            "auction",
            "escalation",
            "projects",
            "playPrefix",
            "state",
            "outcome",
            "tactics",
            "proof",
        ]
    ]
    logicalOperators: list[Literal["all", "any", "not", "count"]]
    hardness: list[Literal["hard", "prefer", "report"]]
    proofModes: list[Literal["exact", "bounded", "policy_conditioned"]]
    proofStrategyClasses: list[Literal["pure_deterministic", "frozen_policy"]]
    proofVisibilities: list[Literal["complete_state", "actor_information_set"]]
    proofClaims: list[
        Literal[
            "outcome_reachable",
            "outcome_forced",
            "winning_root_action_count",
            "winning_continuation_line_count",
            "selected_action_is_unique_winner",
            "target_team_has_no_salvage",
        ]
    ]
    proofUtilities: list[
        Literal[
            "awarded_point_margin",
            "contract_success",
            "round_win",
            "match_win",
            "kabout",
        ]
    ]
    proofBoundaryEvaluators: list[Literal["current_raw_point_margin_v1"]]
    proofLineUniverses: list[
        Literal["all_legal", "optimal_strategy", "policy_conditioned"]
    ]
    proofHorizonUnits: list[Literal["plies", "decisions", "tricks", "round_end"]]
    predicateRegistry: ScenarioPredicateRegistryV1Response
    proofBudgets: ScenarioProofBudgetRegistryV1Response
    defaultPolicyProfiles: dict[
        Literal["north", "west", "south", "east"],
        ScenarioPolicyProfileV1Response
        | ScenarioDeclarativePolicyProfileV1Response,
    ]
    cardOptions: list[ScenarioCardOptionV1Response]
    tacticOptions: list[str]
    projectOptions: list[ScenarioProjectOptionV1Response]
    ruleOptions: ScenarioRuleOptionsV1Response
    projectDomainHash: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchCapabilityResponse(BaseModel):
    model_config = {"extra": "forbid"}

    recipeSchemaVersion: Literal[4]
    supportedRecipeSchemaVersions: list[Literal[1, 2, 3, 4]]
    recipeConfiguration: dict[str, Any]
    scenarioConfiguration: ResearchScenarioConfigurationV1Response
    decisionAuditSchemaVersion: Literal[2]
    solver: ResearchPluginDescriptorResponse
    auctionCompiler: ResearchPluginDescriptorResponse | None
    escalationCompiler: ResearchPluginDescriptorResponse | None
    playCompiler: ResearchPluginDescriptorResponse | None
    projectCompiler: ResearchPluginDescriptorResponse | None
    solverAvailable: bool
    generationLanes: list[Literal["ortools", "organic_runtime"]]
    auctionConstructionModes: list[str]
    playConstructionModes: list[str]
    recommendedPlayConstructionMode: str
    sharding: ResearchCapabilityShardingResponse
    analysisModes: list[str]
    deepVerificationPolicies: list[str]
    defaultAgentRelease: str
    analysisProfiles: dict[str, str]
    defaultScoreProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    defaultScoreProfile: dict[str, Any]
    defaultDecisionScoreProfileHash: str | None
    defaultDecisionScoreProfile: dict[str, Any] | None
    compilerQualification: ResearchCompilerQualificationResponse
    comparisonProfile: dict[str, Any]
    outputModes: ResearchOutputModesCapabilityResponse
    truthLevels: list[str]
    tacticalVerification: ResearchTacticalVerificationCapabilityResponse
    configurationSelection: ResearchConfigurationSelectionReceiptResponse | None = None
    scenarioResearchV4: ScenarioResearchV4CapabilityV1Response | None = None


class ResearchDomainCapabilitiesResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[2]
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    conventionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    localizationHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rulePrimitives: list[ResearchRulePrimitiveResponse]
    tacticalPatterns: list[str]
    pluginCapabilities: list[ResearchPluginDescriptorResponse]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compilerV3: ResearchDomainCompilerV3Response | None
    research: ResearchCapabilityResponse
    rules: dict[str, Any] | None = None
    conventions: dict[str, Any] | None = None
    localization: dict[str, Any] | None = None


class ResearchRulePrimitiveListResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    domainHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[ResearchRulePrimitiveResponse]
    configurationSelection: ResearchConfigurationSelectionReceiptResponse | None = None


class ResearchSemanticDisagreementV1Response(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1]
    disagreementId: str = Field(pattern=r"^DSG-[A-Z0-9]{12}$")
    runCode: str = Field(pattern=r"^RUN-[A-Z0-9]{12}$")
    candidateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    category: Literal[
        "constraint_mismatch",
        "legality_mismatch",
        "score_mismatch",
        "optimality_mismatch",
        "provenance_mismatch",
    ]
    solverEvidence: dict[str, Any]
    engineEvidence: dict[str, Any]
    status: Literal["open", "resolved", "dismissed"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveCategory: Literal["historical_record_requires_readjudication"]
    semanticDefect: Literal[False]


class ResearchSemanticDisagreementV2Response(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[2]
    disagreementId: str = Field(pattern=r"^DSG-[A-Z0-9]{12}$")
    runCode: str = Field(pattern=r"^RUN-[A-Z0-9]{12}$")
    candidateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    category: Literal[
        "compiler_parity_defect",
        "runtime_transition_defect",
        "hard_constraint_violation",
        "observer_information_defect",
    ]
    semanticDefect: Literal[True]
    solverEvidence: dict[str, Any]
    runtimeEvidence: dict[str, Any]
    status: Literal["open", "resolved", "dismissed"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    effectiveCategory: Literal[
        "compiler_parity_defect",
        "runtime_transition_defect",
        "hard_constraint_violation",
        "observer_information_defect",
    ]


ResearchSemanticDisagreementWireResponse = Annotated[
    ResearchSemanticDisagreementV1Response | ResearchSemanticDisagreementV2Response,
    Field(discriminator="schemaVersion"),
]


class ResearchSemanticDisagreementListResponse(BaseModel):
    model_config = {"extra": "forbid"}

    items: list[ResearchSemanticDisagreementWireResponse]


class ResearchPlayCompilerVerificationArtifactResponse(BaseModel):
    model_config = {"extra": "forbid"}

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)
    mediaType: Literal["application/vnd.balootx.cp-sat-round-parity-v2+json"]
    name: str = Field(min_length=1)


class ResearchPlayCompilerVerificationResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[2]
    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    outcome: Literal["matched"]
    solverStatus: Literal["optimal", "feasible"]
    projectSolverStatus: Literal["optimal"]
    projectPlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    projectPoints: tuple[int, int]
    balootPoints: tuple[int, int]
    projectRevealCount: int = Field(ge=0)
    playPlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidenceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: ResearchPlayCompilerVerificationArtifactResponse


class ResearchSequentialContinuationStatusDistributionResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    optimal: int = Field(ge=0)
    infeasible: int = Field(ge=0)

    @field_validator("*", mode="before")
    @classmethod
    def exact_integers(cls, value):
        if type(value) is not int:
            raise ValueError("sequential receipt distribution values must be integers")
        return value


class ResearchSequentialStopReasonDistributionResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    candidates_exhausted: int = Field(ge=0)
    maximum_solutions_reached: int = Field(ge=0)
    qualified: int = Field(ge=0)
    incomplete: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    fatal_quarantine: int = Field(ge=0)

    @field_validator("*", mode="before")
    @classmethod
    def exact_integers(cls, value):
        if type(value) is not int:
            raise ValueError("sequential receipt distribution values must be integers")
        return value


class ResearchSequentialSearchReceiptSummaryResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    admittedSeedCount: int = Field(ge=0)
    exactRejectionCount: int = Field(ge=0)
    noGoodAppliedCount: int = Field(ge=0)
    noGoodUnappliedCount: int = Field(ge=0)
    continuationSolveCount: int = Field(ge=0)
    continuationStatusDistribution: (
        ResearchSequentialContinuationStatusDistributionResponse
    )
    stopReasonDistribution: ResearchSequentialStopReasonDistributionResponse

    @field_validator(
        "admittedSeedCount",
        "exactRejectionCount",
        "noGoodAppliedCount",
        "noGoodUnappliedCount",
        "continuationSolveCount",
        mode="before",
    )
    @classmethod
    def exact_integers(cls, value):
        if type(value) is not int:
            raise ValueError("sequential receipt summary values must be integers")
        return value


class ResearchSequentialSearchReceiptResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    occurrenceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receiptHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runCode: str = Field(pattern=r"^RUN-[A-Z0-9]{12}$")
    shardId: str = Field(min_length=1)
    attemptCount: int = Field(ge=1)
    fencingToken: int = Field(ge=1)
    seed: int = Field(ge=0, le=(1 << 63) - 1)
    admitted: Literal[True]
    createdAt: str = Field(min_length=1)
    summary: ResearchSequentialSearchReceiptSummaryResponse
    receiptBundle: dict[str, Any]

    @field_validator(
        "schemaVersion",
        "attemptCount",
        "fencingToken",
        "seed",
        mode="before",
    )
    @classmethod
    def exact_integers(cls, value):
        if type(value) is not int:
            raise ValueError("sequential receipt identity values must be integers")
        return value

    @field_validator("admitted", mode="before")
    @classmethod
    def exact_admitted_flag(cls, value):
        if type(value) is not bool:
            raise ValueError("sequential receipt admitted must be a boolean")
        return value


class ResearchSequentialSearchReceiptListResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    items: list[ResearchSequentialSearchReceiptResponse]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    nextOffset: int | None = Field(None, ge=0)
    hasMore: bool

    @field_validator("total", "offset", "limit", "nextOffset", mode="before")
    @classmethod
    def exact_integers(cls, value):
        if value is not None and type(value) is not int:
            raise ValueError("sequential receipt page values must be integers")
        return value

    @field_validator("hasMore", mode="before")
    @classmethod
    def exact_has_more_flag(cls, value):
        if type(value) is not bool:
            raise ValueError("sequential receipt hasMore must be a boolean")
        return value


class ResearchRunLaunchAdmissionResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    recipeHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimate: ResearchRecipeEstimateResponse
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRunDiversityRejectionResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    resultCode: str = Field(pattern=r"^(?:RES-[A-Z0-9]{12}|LAF-[A-Z0-9]{10})$")
    reason: Literal["duplicate_fingerprint", "diversity_or_result_limit", "interest_gate_rejected"]


class ResearchRunDiversitySelectionResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    profileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested: int = Field(ge=1, le=1000)
    qualifiedCount: int = Field(ge=0)
    selectedCount: int = Field(ge=0)
    selectedResultCodes: list[str]
    rejected: list[ResearchRunDiversityRejectionResponse]
    rejectedAuditSample: list[str]
    cellCounts: dict[str, int]
    duplicateCount: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRunOutputReceiptResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    recipeHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output: ResearchOutputV3Request
    renderPolicy: ResearchRenderPolicyV4Request
    requestedResultCount: int = Field(ge=1, le=1000)
    requestedVideoCount: int = Field(ge=0, le=100)
    retainedResultCount: int = Field(ge=0)
    qualifiedResultCount: int = Field(ge=0)
    selectedResultCount: int = Field(ge=0)
    resultTargetSatisfied: bool
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchScenarioYieldCountsResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    attempted: int = Field(ge=0)
    structurallyFeasible: int = Field(ge=0)
    legalReplay: int = Field(ge=0)
    semanticSatisfied: int = Field(ge=0)
    proofCompleted: int = Field(ge=0)
    qualified: int = Field(ge=0)


class ResearchRunErrorResponse(BaseModel):
    model_config = {"extra": "ignore", "strict": True}

    type: str = Field(min_length=1)
    message: str


ResearchSemanticStatusV1 = Literal[
    "satisfied",
    "structurally_infeasible",
    "not_found_within_budget",
    "semantic_postcondition_miss",
    "proof_incomplete",
    "proof_failed",
]


class ResearchRunResponse(BaseModel):
    model_config = {"extra": "ignore", "strict": True}
    schemaVersion: Literal[1] = 1
    runCode: str = Field(pattern=r"^RUN-[A-Z0-9]{12}$")
    runHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    recipeHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configurationAuthority: ResearchConfigurationAuthorityResponse | None = None
    researchContextHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    generationLane: Literal["ortools", "organic_runtime", "exact_input"] | None = None
    dealGeneratorIdentity: dict[str, Any] | None = None
    dealGeneratorDescriptorHash: str | None = Field(
        None, pattern=r"^[0-9a-f]{64}$"
    )
    comparisonCampaignCode: str | None = None
    scenarioExecutorIdentityHash: str | None = Field(
        None, pattern=r"^[0-9a-f]{64}$"
    )
    scenarioExecutorIdentity: dict[str, Any] | None = None
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0.0, le=1.0)
    resultCount: int = Field(0, ge=0)
    qualifiedCount: int = Field(0, ge=0)
    selectedCount: int | None = Field(None, ge=0)
    duplicateQualifiedCount: int | None = Field(None, ge=0)
    rejectedCount: int | None = Field(None, ge=0)
    quarantineCount: int = Field(0, ge=0)
    requestedResultCount: int | None = Field(None, ge=1, le=1000)
    requestedVideoCount: int | None = Field(None, ge=0, le=100)
    launchAdmission: ResearchRunLaunchAdmissionResponse | None = None
    resultCodes: list[str] | None = None
    auditResultCount: int = Field(0, ge=0)
    auditResultCodes: list[str] | None = None
    attemptedCandidates: int | None = Field(None, ge=0)
    scenarioSearchStatus: Literal[
        "satisfied", "structurally_infeasible", "not_found_within_budget"
    ] | None = None
    scenarioResearchStatus: ResearchSemanticStatusV1 | None = None
    rejectionStatusCounts: dict[str, int] | None = None
    constructionReceiptHashes: list[str] | None = None
    noGoodReceiptHashes: list[str] | None = None
    scenarioYieldCounts: ResearchScenarioYieldCountsResponse | None = None
    scenarioSearchEvidence: dict[str, Any] | None = None
    scenarioResearchEvidence: dict[str, Any] | None = None
    partialPerformanceReceipt: dict[str, Any] | None = None
    performanceReceipt: dict[str, Any] | None = None
    campaignProgress: dict[str, Any] | None = None
    campaignAccounting: dict[str, Any] | None = None
    diversitySelection: ResearchRunDiversitySelectionResponse | None = None
    outputReceipt: ResearchRunOutputReceiptResponse | None = None
    error: ResearchRunErrorResponse | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    auctionOpenings: list[ResearchAuctionSeedV1Request] | SkipJsonSchema[None] = None
    createdAt: str
    startedAt: str | None = None
    completedAt: str | None = None
    updatedAt: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_read_shape(cls, value):
        if isinstance(value, Mapping) and "optimalityCutCount" in value:
            value = {**value}
            value.pop("optimalityCutCount")
        return value


class ResearchLibraryProjectionResponse(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    actingSeats: list[Literal["north", "west", "south", "east"]]
    observerSeats: list[Literal["north", "west", "south", "east"]]
    contract: Literal["sun", "hokum", "ashkal"] | None = None
    purchaseRound: Literal[1, 2] | None = None
    trumpSuit: Literal["spades", "hearts", "diamonds", "clubs"] | None = None
    escalation: str = "normal"
    scorePressure: str | None = None
    projectKinds: list[str]
    generationLane: Literal["ortools", "organic_runtime", "exact_input"]


class ResearchStrategicSelectionSummaryResponse(BaseModel):
    schemaVersion: Literal[1] = 1
    profileHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    decisionCount: int = Field(ge=0)
    exactDecisionCount: int = Field(ge=0)
    provisionalDecisionCount: int = Field(ge=0)
    fallbackDecisionCount: int = Field(ge=0)
    truthLevels: list[str]
    proofScope: str | None = None
    globalOptimalityClaimed: Literal[False] = False


class ScenarioPredicateEvidenceV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    predicateId: str
    predicateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str
    hardness: Literal["hard", "prefer", "report"]
    status: Literal["satisfied", "failed", "not_evaluated"]
    constructionCapability: Literal[
        "cp_sat_exact",
        "incremental_actor_safe",
        "guided_search",
        "postcondition_only",
        "exact_input_only",
        "unsupported",
    ]
    runtimeVerifier: str
    observed: dict[str, Any]
    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    failureCode: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioSatisfactionV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    scenarioHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    constraintsSatisfied: bool
    outcomeSatisfied: bool
    hardPredicatesSatisfied: bool
    status: ResearchSemanticStatusV1
    predicates: list[ScenarioPredicateEvidenceV1Response]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioConstraintPredicateEvidenceV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    predicateId: str
    predicateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    constructionCapability: Literal[
        "cp_sat_exact",
        "incremental_actor_safe",
        "guided_search",
        "postcondition_only",
        "exact_input_only",
        "unsupported",
    ]
    constructionStatus: Literal[
        "exact_input_replayed",
        "constructed_and_replayed",
        "not_constructed",
    ]
    runtimeEvidenceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtimeStatus: Literal["satisfied", "failed", "not_evaluated"]


class ScenarioConstraintEvidenceV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    scenarioHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilityPlanHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["search", "exact_trace_prefix"]
    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    exactStateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicateEvidence: list[ScenarioConstraintPredicateEvidenceV1Response]
    infeasibilityCore: list[str] | None = None
    constructionStatus: str
    constructionReceipt: dict[str, Any] | None = None
    runtimeStatus: Literal["satisfied", "semantic_postcondition_miss"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioContinuationLinesV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    complete: bool
    exactCount: str | None = Field(None, pattern=r"^[0-9]+$")
    lowerBound: str | None = Field(None, pattern=r"^[0-9]+$")
    upperBound: str | None = Field(None, pattern=r"^[0-9]+$")
    lineUniverse: Literal["all_legal", "optimal_strategy", "policy_conditioned"]


class ScenarioProofOutcomeV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    requestedMode: Literal["exact", "bounded", "policy_conditioned"]
    achievedMode: Literal["exact", "bounded", "policy_conditioned", "none"]
    requestedVisibility: Literal["complete_state", "actor_information_set"]
    achievedVisibility: Literal["complete_state", "actor_information_set", "none"]
    strategyClass: Literal["pure_deterministic", "frozen_policy"] | None = None
    claim: Literal[
        "outcome_reachable",
        "outcome_forced",
        "winning_root_action_count",
        "winning_continuation_line_count",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
    ]
    targetTeam: Literal["north_south", "west_east"]
    utility: str
    successThreshold: float
    horizon: dict[str, Any] | None = None
    boundaryEvaluator: str | None = None
    rootActions: list[dict[str, Any]]
    selectedAction: str | None = None
    winningRootActionCount: int | None = Field(default=None, ge=0)
    winningRootActionLowerBound: int = Field(ge=0)
    winningRootActionUpperBound: int | None = Field(default=None, ge=0)
    continuationLines: ScenarioContinuationLinesV1Response
    hiddenWorlds: dict[str, Any]
    resources: dict[str, Any]
    completionStatus: Literal[
        "exact_complete",
        "bounded_complete",
        "policy_complete",
        "proof_incomplete",
        "proof_failed",
    ]
    claimSatisfied: bool
    principalVariation: list[str]
    identities: dict[str, Any]
    semanticSha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    artifactSha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchLessonTagsV1Response(BaseModel):
    """Editorial tag view without private proof or normalization identities."""
    model_config = {"extra": "forbid", "strict": True}
    schemaVersion: Literal[1]
    tags: dict[str, list[str]] = Field(json_schema_extra={
        'additionalProperties': False,
        'properties': {axis: {'type': 'array', 'uniqueItems': True,
            'maxItems': len(values), 'items': {'type': 'string', 'enum': list(values)}}
            for axis, values in TAG_AXES.items()}})
    playIndex: int = Field(ge=16, le=27)
    qualificationEvidence: Literal[False]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator('tags')
    @classmethod
    def validate_tag_values(cls, value):
        if set(value) - set(TAG_AXES):
            raise ValueError('lesson tag axes are invalid')
        TagQueryV1.build({axis: values for axis, values in value.items() if values})
        return value

    @model_validator(mode='after')
    def validate_public_tag_hash(self):
        if canonical_sha256(self.model_dump(exclude={'sha256'})) != self.sha256:
            raise ValueError('public lesson tag hash mismatch')
        return self


class ScenarioProjectionV1Response(BaseModel):
    model_config = {"extra": "forbid"}
    schemaVersion: Literal[1] = 1
    scenarioTemplateId: str = Field(pattern=r"^SCN-[A-F0-9]{12}$")
    origin: Literal["search", "exact_trace_prefix"]
    status: Literal["satisfied", "semantic_postcondition_miss"]
    decisionKind: Literal["bid", "escalation", "project", "play"]
    targetTeam: Literal["north_south", "west_east"]
    requestedProofMode: Literal["exact", "bounded", "policy_conditioned"]
    achievedProofMode: Literal["exact", "bounded", "policy_conditioned", "none"]
    proofCompletion: Literal[
        "exact_complete",
        "bounded_complete",
        "policy_complete",
        "proof_incomplete",
        "proof_failed",
    ]
    proofVisibility: Literal["complete_state", "actor_information_set", "none"]
    winningRootActionCount: int | None = Field(default=None, ge=0)
    winningRootActionLowerBound: int = Field(ge=0)
    winningRootActionUpperBound: int | None = Field(default=None, ge=0)
    contractOutcome: Literal["won", "lost"] | None = None
    projectKinds: list[str]
    projectValues: list[str]
    projectStates: list[str]
    scenarioCategories: list[
        Literal[
            "deal",
            "auction",
            "escalation",
            "projects",
            "playPrefix",
            "state",
            "outcome",
            "tactics",
            "proof",
        ]
    ] = Field(default_factory=list)
    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    exactStateAvailable: Literal[True]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LegacyResearchResultResponse(BaseModel):
    model_config = {"extra": "allow"}
    schemaVersion: Literal[1] = 1
    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    resultHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runCode: str = Field(pattern=r"^RUN-[A-Z0-9]{12}$")
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    traceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gameHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float = Field(ge=0.0, le=100.0)
    scoreEvidence: dict[str, Any]
    truthLabel: Literal[
        "exact_optimal", "robust_best_known", "debatable", "provisional", "rejected"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    disposition: Literal["qualified", "rejected", "quarantined"]
    qualification: dict[str, Any]
    classification: str
    criticalDecisionIds: list[str]
    interestingSeats: list[Literal["north", "west", "south", "east"]]
    tacticalTags: list[str]
    verifiedTacticalTags: list[str]
    decisionCount: int = Field(ge=1)
    strategicSelection: ResearchStrategicSelectionSummaryResponse | None = None
    scenarioSatisfaction: ScenarioSatisfactionV1Response | None = None
    constraintEvidence: ScenarioConstraintEvidenceV1Response | None = None
    proofOutcome: ScenarioProofOutcomeV1Response | None = None
    scenarioProjection: ScenarioProjectionV1Response | None = None
    artifacts: dict[str, Any]
    provenance: dict[str, Any]
    libraryProjection: ResearchLibraryProjectionResponse | None = None
    createdAt: str


class ResearchDecisionReferenceV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    decisionOrdinal: Literal[1] = 1


class ScenarioActorPublicProofOutcomeV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    requestedMode: Literal["exact", "bounded", "policy_conditioned"]
    achievedMode: Literal["exact", "bounded", "policy_conditioned", "none"]
    requestedVisibility: Literal["complete_state", "actor_information_set"]
    achievedVisibility: Literal["complete_state", "actor_information_set", "none"]
    strategyClass: Literal["pure_deterministic", "frozen_policy"] | None = None
    claim: Literal[
        "outcome_reachable",
        "outcome_forced",
        "winning_root_action_count",
        "winning_continuation_line_count",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
    ]
    targetTeam: Literal["north_south", "west_east"]
    utility: str
    successThreshold: float
    horizon: dict[str, Any] | None = None
    boundaryEvaluator: str | None = None
    rootActions: list[dict[str, Any]]
    selectedAction: str | None = None
    winningRootActionCount: int | None = Field(default=None, ge=0)
    winningRootActionLowerBound: int = Field(ge=0)
    winningRootActionUpperBound: int | None = Field(default=None, ge=0)
    continuationLines: ScenarioContinuationLinesV1Response
    hiddenWorlds: dict[str, Any]
    resources: dict[str, Any]
    completionStatus: Literal[
        "exact_complete",
        "bounded_complete",
        "policy_complete",
        "proof_incomplete",
        "proof_failed",
    ]
    claimSatisfied: bool
    identities: dict[str, Any]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioActorPublicProjectionV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    scenarioTemplateId: str = Field(pattern=r"^SCN-[A-F0-9]{12}$")
    origin: Literal["search", "exact_trace_prefix"]
    status: ResearchSemanticStatusV1
    decisionKind: Literal["bid", "escalation", "project", "play"]
    targetTeam: Literal["north_south", "west_east"]
    requestedProofMode: Literal["exact", "bounded", "policy_conditioned"]
    achievedProofMode: Literal["exact", "bounded", "policy_conditioned", "none"]
    proofCompletion: Literal[
        "exact_complete",
        "bounded_complete",
        "policy_complete",
        "proof_incomplete",
        "proof_failed",
    ]
    proofVisibility: Literal["complete_state", "actor_information_set", "none"]
    winningRootActionCount: int | None = Field(default=None, ge=0)
    winningRootActionLowerBound: int = Field(ge=0)
    winningRootActionUpperBound: int | None = Field(default=None, ge=0)
    contractOutcome: Literal["won", "lost"] | None = None
    projectKinds: list[str]
    projectValues: list[str]
    projectStates: list[str]
    scenarioCategories: list[
        Literal[
            "deal",
            "auction",
            "escalation",
            "projects",
            "playPrefix",
            "state",
            "outcome",
            "tactics",
            "proof",
        ]
    ] = Field(default_factory=list)
    exactStateAvailable: bool
    decisionReference: ResearchDecisionReferenceV1Response | None = None
    predicateEvidenceCount: int = Field(ge=0)
    publicProjectionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lessonTags: ResearchLessonTagsV1Response | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode='wrap')
    def omit_absent_lesson_tags(self, handler):
        value = handler(self)
        if self.lessonTags is None:
            value.pop('lessonTags', None)
        return value


class ResearchScenarioPublicResultResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[2] = 2
    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    runCode: str = Field(pattern=r"^RUN-[A-Z0-9]{12}$")
    recipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    score: float = Field(ge=0.0, le=100.0)
    truthLabel: Literal["exact_optimal", "robust_best_known", "rejected"]
    confidence: float = Field(ge=0.0, le=1.0)
    disposition: Literal["qualified", "rejected"]
    qualification: dict[str, Any]
    classification: Literal["scenario_v4"]
    interestingSeats: list[Literal["north", "west", "south", "east"]]
    tacticalTags: list[str]
    verifiedTacticalTags: list[str]
    decisionCount: int = Field(ge=0)
    libraryProjection: ResearchLibraryProjectionResponse
    proofOutcome: ScenarioActorPublicProofOutcomeV1Response
    scenarioProjection: ScenarioActorPublicProjectionV1Response
    decisionReference: ResearchDecisionReferenceV1Response | None = None
    publicProjectionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    researchBackend: Literal["ortools_cp_sat_v2", "ortools_cp_sat_v3"] | None = None
    qualificationAuthority: Literal["cp_sat"] | None = None
    runtimeCompatibilityStatus: Literal["not_checked", "passed", "failed"] | None = None
    createdAt: str


ResearchSha256V1 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ResearchAuthorityV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    backendId: Literal["ortools_cp_sat_v2", "ortools_cp_sat_v3"]
    backendVersion: str = Field(min_length=1)
    qualificationAuthority: Literal["cp_sat"]
    solverVersion: str = Field(min_length=1)
    bundleRevisionId: str = Field(min_length=1, max_length=200)
    bundleHash: ResearchSha256V1
    domainIrHashes: dict[str, ResearchSha256V1]
    loweringManifestHash: ResearchSha256V1
    constructionModelHash: ResearchSha256V1
    proofModelHash: ResearchSha256V1
    graphHash: ResearchSha256V1
    traceHash: ResearchSha256V1
    proofOutcomeHash: ResearchSha256V1
    sha256: ResearchSha256V1

    @field_validator("domainIrHashes")
    @classmethod
    def validate_domain_hashes(cls, value):
        if not value:
            raise ValueError("researchAuthority domainIrHashes must be non-empty")
        return dict(sorted(value.items()))


class RuntimeCompatibilityV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    status: Literal["not_checked", "passed", "failed"]
    engineVersion: str | None = None
    bundleHash: ResearchSha256V1
    traceHash: ResearchSha256V1
    evidenceHash: ResearchSha256V1 | None = None
    failureCodes: list[str]
    sha256: ResearchSha256V1


class ExecutionReceiptV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    schemaVersion: Literal[1] = 1
    semanticResultHash: ResearchSha256V1
    stageTimings: dict[str, float | int]
    cpuSeconds: float | int = Field(ge=0)
    peakMemoryBytes: int = Field(ge=0)
    conflicts: int = Field(ge=0)
    branches: int = Field(ge=0)
    cacheBytes: int = Field(ge=0)
    completedVariants: int = Field(ge=0)
    machine: dict[str, JsonValue]
    sha256: ResearchSha256V1

    @field_validator("stageTimings")
    @classmethod
    def validate_stage_timings(cls, value):
        if any(duration < 0 for duration in value.values()):
            raise ValueError("executionReceipt stageTimings must be non-negative")
        return dict(sorted(value.items()))


class ResearchOperatorAuditV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    researchAuthority: ResearchAuthorityV1Response
    runtimeCompatibility: RuntimeCompatibilityV1Response
    executionReceipt: ExecutionReceiptV1Response


class ResearchPlayingQualityDecisionV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True, "allow_inf_nan": False}

    ordinal: int = Field(ge=1)
    kind: Literal["bid", "escalation", "project", "play"]
    actor: Literal["north", "east", "south", "west"]
    action: str = Field(min_length=1)
    legalActionCount: int = Field(ge=1)
    actualActionOptimal: bool
    regret: float | int = Field(ge=0)
    selectionReason: str = Field(min_length=1)
    proofMode: Literal["exact", "bounded"]
    proofVisibility: Literal["complete_state", "actor_information_set"]
    proofCompletionStatus: Literal["exact_complete", "bounded_complete"]
    proofScope: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    proofArtifactHash: ResearchSha256V1
    actionQualityHash: ResearchSha256V1
    sha256: ResearchSha256V1


class ResearchFeaturedPlayHiddenWorldsV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    method: Literal["supplied_worlds_only"]
    count: int = Field(ge=1)
    complete: Literal[False]


class ResearchFeaturedPlayExactHiddenWorldsV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    method: Literal["exhaustive_cp_sat_hidden_allocations_v1"]
    count: int = Field(ge=1)
    complete: Literal[True]


class ResearchFeaturedPlayProofV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    playIndex: int = Field(ge=0, le=31)
    actor: Literal["north", "east", "south", "west"]
    action: str = Field(min_length=1)
    proofArtifactHash: ResearchSha256V1
    proofModelKind: Literal["native_observed_root_outcome_v1", "native_complete_state_root_outcome_v1", "actor_information_set_cegis_model_v1"]
    completionStatus: Literal["bounded_complete", "exact_complete"]
    hiddenWorlds: ResearchFeaturedPlayHiddenWorldsV1Response | ResearchFeaturedPlayExactHiddenWorldsV1Response | None = None

    @model_validator(mode="after")
    def require_matching_proof_scope(self):
        complete_state = self.proofModelKind == "native_complete_state_root_outcome_v1"
        exact_actor = self.proofModelKind == "actor_information_set_cegis_model_v1"
        if self.completionStatus != ("exact_complete" if complete_state or exact_actor else "bounded_complete") or (
                (self.hiddenWorlds is None) != complete_state) or (
                self.hiddenWorlds is not None and self.hiddenWorlds.complete != exact_actor):
            raise ValueError("featured proof completion and visibility scope disagree")
        return self


class ResearchPlayingQualityAuditV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    resultHash: ResearchSha256V1
    scope: Literal["all_decisions"]
    status: Literal["complete"]
    wholeRoundCertified: Literal[False] = False
    operation: Literal["generate", "audit_recorded", "audit_then_generate"]
    qualityReceiptHash: ResearchSha256V1
    completeTraceHash: ResearchSha256V1
    decisionCount: int = Field(ge=0)
    optimalDecisionCount: int = Field(ge=0)
    zeroRegretDecisionCount: int = Field(ge=0)
    countsByKind: dict[Literal["bid", "escalation", "project", "play"], int]
    decisions: list[ResearchPlayingQualityDecisionV1Response]
    featuredPlayProof: ResearchFeaturedPlayProofV1Response | None = None
    sha256: ResearchSha256V1

    @model_validator(mode="after")
    def validate_counts(self):
        if (
            set(self.countsByKind) != {"bid", "escalation", "project", "play"}
            or any(value < 0 for value in self.countsByKind.values())
            or sum(self.countsByKind.values()) != self.decisionCount
            or len(self.decisions) != self.decisionCount
            or self.optimalDecisionCount
            != sum(item.actualActionOptimal for item in self.decisions)
            or self.zeroRegretDecisionCount
            != sum(item.regret == 0 for item in self.decisions)
        ):
            raise ValueError("playing-quality audit decision counts do not match")
        return self


ResearchResultResponse = Annotated[
    LegacyResearchResultResponse | ResearchScenarioPublicResultResponse,
    Field(discriminator="schemaVersion"),
]


def _public_research_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if result.get("schemaVersion") == 2:
        proof = dict(result.get("proofOutcome") or {})
        authority = dict(result.get("researchAuthority") or {})
        if (
            authority.get("backendId") == "ortools_cp_sat_v3"
            and proof.get("completionStatus") == "proof_incomplete"
            and proof.get("rootActions")
        ):
            document, owner_hash = _load_owned_result_json_artifact_v1(
                RESEARCH.artifact_store,
                dict(dict(result.get("artifacts") or {}).get("proofOutcome") or {}),
                media_type=ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES["proofOutcome"],
                owner_kind="ortools_research_semantic_artifact_v1",
            )
            identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
            if (
                owner_hash != identity.get("executorSemanticHash")
                or canonical_sha256(document) != authority.get("proofOutcomeHash")
            ):
                raise ValueError("Research public proof artifact authority mismatch")
            if not document.get("rootActions"):
                # Historical receipts synthesized rows even when no values were solved.
                result["proofOutcome"] = {**proof, "rootActions": []}
        public = public_scenario_result_v4(result)
        sections = (
            result.get("researchAuthority"),
            result.get("runtimeCompatibility"),
            result.get("executionReceipt"),
        )
        if any(section is not None for section in sections):
            if any(not isinstance(section, Mapping) for section in sections):
                raise ValueError("Research V4 authority sections are incomplete")
            authority = ResearchAuthorityContractV1.from_dict(sections[0]).to_dict()
            compatibility = RuntimeCompatibilityContractV1.from_dict(
                sections[1]
            ).to_dict()
            ExecutionReceiptContractV1.from_dict(sections[2])
            public.update(
                {
                    "researchBackend": authority["backendId"],
                    "qualificationAuthority": authority["qualificationAuthority"],
                    "runtimeCompatibilityStatus": compatibility["status"],
                }
            )
        return public
    return result


class ResearchDecisionAuditResponse(BaseModel):
    model_config = {"extra": "allow"}
    schemaVersion: Literal[1] = 1
    decisionId: str
    ordinal: int = Field(ge=1)
    decisionOrdinal: int | None = Field(None, ge=0)
    actingSeat: Literal["north", "west", "south", "east"]
    actingTeam: Literal["north_south", "west_east"]
    decisionKind: Literal["bid", "play", "project", "escalation"]
    phase: str
    publicStateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    informationStateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    legalActions: list[dict[str, Any]]
    rulePaths: list[str]
    actualAction: dict[str, Any]
    actionSource: str
    recommendedAction: dict[str, Any]
    recommendationAvailable: bool = True
    actualMatchesRecommendation: bool
    forced: bool
    alternatives: list[dict[str, Any]]
    expectedRegret: float
    probabilityDrop: float
    catastrophicCrossing: bool
    hiddenDealSupport: int = Field(ge=0)
    effectiveSampleSize: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    convergence: str
    principalContinuation: list[Any]
    supportedMetrics: list[str]
    unsupportedMetrics: list[str]
    selectionExplanation: dict[str, Any]
    analyticalExplanation: dict[str, Any]
    forcedExplanation: dict[str, Any] | None
    tacticalEvidence: list[dict[str, Any]]
    actualDealOracle: dict[str, Any] | None
    truthLabel: Literal[
        "exact_optimal", "robust_best_known", "debatable", "provisional", "rejected"
    ]
    eventSequence: int | None = None
    eventHash: str | None = None
    provenance: dict[str, Any]


class ResearchRecipeListResponse(BaseModel):
    model_config = {"extra": "forbid"}

    items: list[StoredResearchRecipeWireResponse]
    nextCursor: str | None


class ResearchMutationErrorResponse(BaseModel):
    model_config = {"extra": "forbid"}

    detail: str


class ResearchRequestValidationIssueResponse(BaseModel):
    model_config = {"extra": "allow"}

    loc: list[str | int]
    msg: str
    type: str


class ResearchRequestValidationErrorResponse(BaseModel):
    model_config = {"extra": "forbid"}

    detail: list[ResearchRequestValidationIssueResponse]


class ResearchMutationValidationErrorResponse(
    RootModel[ResearchMutationErrorResponse | ResearchRequestValidationErrorResponse]
):
    pass


class ResearchRunConflictResponse(
    RootModel[ResearchMutationErrorResponse | ResearchLaunchAdmissionErrorResponse]
):
    pass


class ResearchRunListResponse(BaseModel):
    items: list[ResearchRunResponse]


class ResearchTimingDistributionResponse(BaseModel):
    sampleCount: int = Field(ge=0)
    p50Seconds: float | None = Field(None, ge=0.0)
    p95Seconds: float | None = Field(None, ge=0.0)
    totalSeconds: float | None = Field(None, ge=0.0)
    percentileMethod: Literal["linear_interpolation_v1"]
    sampleUnit: Literal[
        "completed_shard_stage_total",
        "completed_shard_elapsed_to_milestone",
        "completed_shard_calling_thread",
    ]


class ResearchCpuMeasurementResponse(ResearchTimingDistributionResponse):
    availability: Literal["measured", "mixed", "unavailable"]
    measurementBoundary: str | None = None
    nativeWorkOnCallingThreadIncluded: Literal[True]
    auxiliaryThreadsExcluded: Literal[True]
    note: str


class ResearchCostProxyResponse(BaseModel):
    wallSecondsPerQualified: float | None = Field(None, ge=0.0)
    callingOsThreadCpuSecondsPerQualified: float | None = Field(None, ge=0.0)
    monetaryCost: None = None
    note: str


class ResearchOperationsStatsResponse(BaseModel):
    currentQueuedShardCount: int = Field(ge=0)
    currentRunningShardCount: int = Field(ge=0)
    currentCancelledShardCount: int = Field(ge=0)
    currentFailedShardCount: int = Field(ge=0)
    shardExecutionCount: int = Field(ge=0)
    repeatedExecutionCount: int = Field(ge=0)
    retryCount: int | None = Field(None, ge=0)
    retryAvailability: Literal["measured", "unavailable_historical"]
    reclaimCount: int | None = Field(None, ge=0)
    reclaimAvailability: Literal["measured", "unavailable_historical"]


class ResearchMeasurementAvailabilityItemResponse(BaseModel):
    status: Literal["measured", "mixed", "unavailable", "excluded"]
    reason: str


class ResearchMeasurementAvailabilityResponse(BaseModel):
    model_config = {"extra": "forbid"}

    qualifiedDistinctScenarios: ResearchMeasurementAvailabilityItemResponse
    qualifiedDistinctScenarioWallRate: ResearchMeasurementAvailabilityItemResponse
    qualifiedDistinctScenarioCpuRate: ResearchMeasurementAvailabilityItemResponse
    hardware: ResearchMeasurementAvailabilityItemResponse
    peakMemory: ResearchMeasurementAvailabilityItemResponse
    cache: ResearchMeasurementAvailabilityItemResponse
    nativeChildCpu: ResearchMeasurementAvailabilityItemResponse
    retries: ResearchMeasurementAvailabilityItemResponse
    reclaims: ResearchMeasurementAvailabilityItemResponse


class ResearchEnvironmentEvidenceResponse(BaseModel):
    status: Literal["measured", "mixed", "unavailable"]
    declaration: dict[str, Any] | None = None
    environmentHashes: list[str]
    comparisonHashes: list[str]
    sampleCount: int = Field(ge=0)
    note: str


class ResearchPeakMemoryResponse(BaseModel):
    availability: Literal["measured", "mixed", "unavailable"]
    peakProcessRssBytes: int | None = Field(None, ge=0)
    sampleCount: int = Field(ge=0)
    measurementBoundary: str | None = None
    aggregation: Literal["maximum_observed_process_lifetime_peak_v1"]
    note: str


class ResearchCacheScopeResponse(BaseModel):
    status: Literal["measured"]
    measurementBoundary: Literal["process_lifetime_cumulative_snapshot_v1"]
    aggregation: Literal["maximum_observed_snapshot_not_additive_v1"]
    sampleCount: int = Field(ge=1)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    evictions: int = Field(ge=0)
    entries: int = Field(ge=0)
    capacity: int = Field(ge=1)
    note: str


class ResearchSolverStatusDistributionResponse(BaseModel):
    construction: dict[str, int]
    playCompiler: dict[str, int]
    coverage: Literal["recorded_status_counters_only_v1"]


class ResearchPerformanceRollupResponse(BaseModel):
    schemaVersion: Literal[1] = 1
    shardCount: int = Field(ge=0)
    completedShardCount: int = Field(ge=0)
    attemptedSeeds: int = Field(ge=0)
    resultCount: int = Field(ge=0)
    qualifiedCount: int = Field(ge=0)
    qualifiedDistinctScenarioCount: int | None = Field(ge=0)
    qualifiedDistinctScenariosPerWallHour: float | None = Field(ge=0.0)
    qualifiedDistinctScenariosPerCpuHour: float | None = Field(ge=0.0)
    qualifiedPer100Attempts: float = Field(ge=0.0)
    resultPer100Attempts: float = Field(ge=0.0)
    duplicateCount: int | None = Field(None, ge=0)
    duplicateRate: float | None = Field(None, ge=0.0, le=1.0)
    duplicateEvidence: Literal[
        "complete_result_code_evidence_v1", "unavailable_or_partial"
    ]
    timing: dict[str, ResearchTimingDistributionResponse]
    milestoneTiming: dict[str, ResearchTimingDistributionResponse]
    cpu: ResearchCpuMeasurementResponse
    environment: ResearchEnvironmentEvidenceResponse
    peakMemory: ResearchPeakMemoryResponse
    cacheScopes: dict[str, ResearchCacheScopeResponse]
    costProxy: ResearchCostProxyResponse
    operations: ResearchOperationsStatsResponse
    solverStatusDistribution: ResearchSolverStatusDistributionResponse
    measurementAvailability: ResearchMeasurementAvailabilityResponse


class ResearchStatsResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    scope: Literal["all", "run"]
    runCode: str | None = None
    runStatus: dict[str, int]
    resultDisposition: dict[str, int]
    truth: dict[str, int]
    resultCount: int = Field(ge=0)
    averageScore: float = Field(ge=0.0, le=100.0)
    averageConfidence: float = Field(ge=0.0, le=1.0)
    semanticDefects: dict[str, int]
    historicalRecords: dict[str, int]
    funnel: dict[str, int]
    sequentialSearch: ResearchSequentialSearchReceiptSummaryResponse
    measuredRunCount: int = Field(ge=0)
    cpuMeasuredRunCount: int = Field(ge=0)
    elapsedWallSeconds: float = Field(ge=0.0)
    cpuSeconds: float | None = Field(None, ge=0.0)
    cpuMeasurementBoundary: str | None = None
    cpuMeasurementNote: str | None = None
    qualifiedPerWallHour: float = Field(ge=0.0)
    qualifiedPerCpuHour: float | None = Field(None, ge=0.0)
    qualifiedPer100Attempts: float = Field(ge=0.0)
    stageSeconds: dict[str, float] = Field(default_factory=dict)
    byGenerationLane: dict[str, dict[str, Any]] = Field(default_factory=dict)
    performance: ResearchPerformanceRollupResponse
    performanceByGenerationLane: dict[str, ResearchPerformanceRollupResponse] = Field(
        default_factory=dict
    )


class ComparisonSmokeEvidenceResponse(BaseModel):
    scope: Literal["smoke_only_not_promotion_evidence"]
    minimumQualifiedPerLane: int = Field(ge=1)
    passed: bool
    reasons: list[str]


class ComparisonPromotionEvidenceResponse(BaseModel):
    model_config = {"extra": "allow"}
    schemaVersion: Literal[1, 2]
    profileHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    evidenceStatus: str | None = None
    eligibleForPromotion: bool
    smokeEvidence: ComparisonSmokeEvidenceResponse | None = None
    minimumDistinctSelectedFinalistsPerLane: int | None = Field(None, ge=1)
    minimumAttemptedSeedsPerLane: int | None = Field(None, ge=1)
    requiredPromotionCpuBoundary: str | None = None
    measurementComparable: bool | None = None
    cpuBoundaryComparable: bool | None = None
    yieldMultiplier: float | None = Field(None, ge=0.0)
    costReduction: float | None = None
    yieldGatePassed: bool | None = None
    costGatePassed: bool | None = None
    passed: bool
    recommendation: Literal[
        "promote_ortools", "retain_current", "insufficient_evidence"
    ]
    reasons: list[str]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ComparisonCampaignResponse(BaseModel):
    model_config = {"extra": "allow"}
    schemaVersion: Literal[1] = 1
    campaignCode: str = Field(pattern=r"^CMP-[A-Z0-9]{12}$")
    campaignHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseRecipeCode: str = Field(pattern=r"^RCP-[A-Z0-9]{6,20}$")
    baseRecipeHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generationLanes: list[Literal["ortools", "organic_runtime"]]
    recipeCodes: dict[str, str]
    runCodes: dict[str, str]
    matchedInputs: dict[str, Any]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0.0, le=1.0)
    runs: dict[str, dict[str, Any]]
    comparison: ComparisonPromotionEvidenceResponse
    createdAt: str


class ComparisonCampaignListResponse(BaseModel):
    items: list[ComparisonCampaignResponse]


class ResearchResultGroupCountResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    value: str
    count: int = Field(ge=1)


class ResearchResultListResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    items: list[ResearchResultResponse]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    nextOffset: int | None = Field(None, ge=0)
    hasMore: bool
    groupBy: (
        Literal[
            "tactical_tag",
            "acting_seat",
            "observer_seat",
            "contract",
            "project",
            "generation_lane",
            "truth_label",
            "disposition",
            "scenario_template",
            "scenario_origin",
            "scenario_status",
            "requested_proof_mode",
            "achieved_proof_mode",
            "proof_completion",
            "proof_visibility",
            "target_team",
            "contract_outcome",
            "project_kind",
            "project_value",
            "project_state",
            "winning_root_action_band",
            "exact_state_available",
            "decision_kind",
            "research_backend",
            "qualification_authority",
            "runtime_compatibility_status",
            "configuration_bundle",
            "exact_state_id",
            "scenario_category",
            "variant_collection",
        ]
        | None
    ) = None
    groups: list[ResearchResultGroupCountResponse] = Field(default_factory=list)


class ResearchDecisionListResponse(BaseModel):
    resultCode: str
    items: list[ResearchDecisionAuditResponse]


class ResearchScenarioExactStateViewV1Response(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    decisionReference: ResearchDecisionReferenceV1Response
    publicProjectionHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: str
    decisionKind: Literal["bid", "escalation", "project", "play"]
    observerSeat: Literal["north", "west", "south", "east"]
    nextActor: Literal["north", "west", "south", "east"]
    legalActions: list[dict[str, Any]]
    publicState: dict[str, Any]


class ResearchPerspectiveEvidenceResponse(BaseModel):
    model_config = {"extra": "allow"}

    schemaVersion: Literal[3]
    decisionId: str
    actingSeat: Literal["north", "west", "south", "east"]
    actingTeam: Literal["north_south", "west_east"]
    observerSeat: Literal["north", "west", "south", "east"]
    truthMode: Literal["actor_observer", "seat_experience", "actual_deal_oracle"]
    projectionHash: str
    forced: bool
    truthLabel: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float]
    recommendedAction: dict[str, Any] | None = None
    promptContract: str | None = None


class ResearchPerspectiveListResponse(BaseModel):
    schemaVersion: Literal[1] = 1
    resultCode: str
    observerSeat: Literal["north", "west", "south", "east"]
    items: list[ResearchPerspectiveEvidenceResponse]


class ResearchRenderPrepareRequest(BaseModel):
    model_config = {"extra": "forbid"}

    revealPolicy: Literal["after_commit", "after_completion", "post_reveal_only"] = (
        "after_commit"
    )
    exportPreset: Literal["decision_clip", "hooked_flagship"] | None = None
    observerSeat: Literal["north", "west", "south", "east"] | None = None


class ResearchRenderDecisionSelectionEvidenceResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    profileHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    evidenceHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    decisionId: str | None = Field(None, min_length=1)
    gatePassed: bool
    qualifiedDecision: bool | None = None
    blockers: list[str] | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRenderVideoCandidateEvidenceResponse(BaseModel):
    model_config = {"extra": "forbid"}

    decisionKind: str = Field(min_length=1)
    critical: bool
    truthLabel: str = Field(min_length=1)
    selectionScore: float
    confidence: float = Field(ge=0.0, le=1.0)
    compatibilityBlockers: list[str]


class ResearchAuditRenderVideoCandidateEvidenceResponse(BaseModel):
    model_config = {"extra": "forbid"}

    decisionId: str = Field(min_length=1)
    auditOnly: Literal[True]


class ResearchRenderVideoProductionEvidenceResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    profileHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    candidate: (
        ResearchRenderVideoCandidateEvidenceResponse
        | ResearchAuditRenderVideoCandidateEvidenceResponse
        | None
    ) = None
    gatePassed: bool
    blockers: list[str] | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRenderInterestAdmissionResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    passed: bool
    evidenceHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRenderPublicationPolicyResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    decisionSelection: ResearchRenderDecisionSelectionEvidenceResponse
    videoProduction: ResearchRenderVideoProductionEvidenceResponse
    blockers: list[str]
    qualityEvidenceHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    interestAdmission: ResearchRenderInterestAdmissionResponse | SkipJsonSchema[None] = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def validate_interest_admission(cls, value):
        if isinstance(value, Mapping) and "interestAdmission" in value and value["interestAdmission"] is None:
            raise ValueError("render publication interest admission must be an object")
        return value

    @model_serializer(mode="wrap")
    def omit_absent_interest_admission(self, handler):
        value = handler(self)
        if self.interestAdmission is None:
            value.pop("interestAdmission", None)
        return value


class ResearchRenderSourceProvenanceResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    sourceSchemaVersion: Literal[2, 3] | None = None
    kind: Literal[
        "research_result",
        "research_scenario_result",
        "analysis_candidate",
        "finalist",
        "laffa",
        "research_audit_trace",
    ]
    code: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$",
    )
    decisionOrdinal: int | None = Field(None, ge=1)
    decisionReference: ResearchDecisionReferenceV1Response | None = None
    publicProjectionHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    decisionId: str | None = Field(None, min_length=1)
    exactStateId: str | None = Field(None, min_length=1)
    qrActionable: bool | None = None
    auditTraceCode: str | None = Field(None, pattern=r"^LAF-AUD-[0-9A-F]{10}$")
    recordHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    attemptStatus: (
        Literal[
            "running",
            "completed",
            "retry_scheduled",
            "failed",
            "reclaimed",
            "cancelled",
        ]
        | None
    ) = None
    observerSeat: Literal["north", "west", "south", "east"] | None = None
    traceHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    analysisHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    resultHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    auditHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    analysisProfileHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    scenarioHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    proofHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    constraintEvidenceHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    exactStateHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    analyzerId: str | None = Field(None, min_length=1)
    analyzerVersion: str | None = Field(
        None,
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$",
    )
    domainHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    rulesHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    proofReference: str | None = Field(None, pattern=r"^LAF-[A-Z0-9]{10}$")
    proofCompletionStatus: (
        Literal[
            "exact_complete",
            "bounded_complete",
            "policy_complete",
            "proof_incomplete",
            "proof_failed",
        ]
        | None
    ) = None
    proofClaimSatisfied: bool | None = None
    proofAchievedMode: (
        Literal["exact", "bounded", "policy_conditioned", "incomplete"] | None
    ) = None
    originKind: str | None = Field(None, min_length=1, max_length=100)
    sourceStatus: str | None = Field(None, min_length=1, max_length=100)
    qualificationStatus: (
        Literal["qualified", "provisional", "rejected", "quarantined", "legacy_replay"]
        | None
    ) = None
    auditOnly: bool | None = None
    renderPurpose: Literal["content_candidate", "research_audit"] | None = None
    qualificationFailures: list[str] | None = None
    publicationBlockers: list[str] | None = None
    publicationEligible: bool | None = None
    publicationPolicy: ResearchRenderPublicationPolicyResponse | None = None

    @field_validator("sourceSchemaVersion", "decisionOrdinal", mode="before")
    @classmethod
    def exact_integers(cls, value):
        if value is not None and type(value) is not int:
            raise ValueError("render source identity values must be integers")
        return value

    @field_validator("qrActionable", "proofClaimSatisfied", mode="before")
    @classmethod
    def exact_boolean_flags(cls, value):
        if value is not None and type(value) is not bool:
            raise ValueError("render source boolean values must be booleans")
        return value


class ResearchScenarioAuthorityReceiptResponse(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1] = 1
    audience: Literal["operator_audit"]
    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    resultHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisionOrdinal: Literal[1] = 1
    exactStateId: str = Field(pattern=r"^STATE-[A-F0-9]{20}$")
    stateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    exactStateHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configurationBundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenarioHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proofSemanticSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proofArtifactSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    constraintEvidenceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResearchRenderPrepareResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    resultCode: str
    decisionOrdinal: int = Field(ge=1)
    decisionId: str
    observerSeat: Literal["north", "west", "south", "east"]
    exportPreset: Literal["decision_clip", "hooked_flagship"]
    qualificationStatus: Literal["qualified", "rejected"]
    publicationEligible: bool
    renderPurpose: Literal["content_candidate", "research_audit"]
    qualificationFailures: list[str]
    publicationBlockers: list[str]
    publicationPolicy: ResearchRenderPublicationPolicyResponse
    locator: dict[str, Any] | None = None
    bundleArtifact: dict[str, Any]
    proofArtifact: dict[str, Any] | None = None
    proofArtifactSha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    authorityReceipt: ResearchScenarioAuthorityReceiptResponse | None = None
    basename: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$",
    )
    preview: dict[str, Any]
    source: ResearchRenderSourceProvenanceResponse


class LaffaRenderPrepareResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    laffaCode: str
    decisionOrdinal: int = Field(ge=1)
    decisionId: str
    observerSeat: Literal["north", "west", "south", "east"]
    exportPreset: Literal["decision_clip", "hooked_flagship"]
    qualificationStatus: Literal["legacy_replay"]
    publicationEligible: Literal[False]
    renderPurpose: Literal["research_audit"]
    qualificationFailures: list[str]
    locator: dict[str, Any]
    bundleArtifact: dict[str, Any]
    basename: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$",
    )
    preview: dict[str, Any]
    source: ResearchRenderSourceProvenanceResponse


class ResearchAuditTraceRenderPrepareResponse(BaseModel):
    model_config = {"extra": "forbid"}

    schemaVersion: Literal[1] = 1
    laffaCode: str
    auditTraceCode: str = Field(pattern=r"^LAF-AUD-[0-9A-F]{10}$")
    recordHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    traceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisionOrdinal: int = Field(ge=1)
    decisionId: str
    observerSeat: Literal["north", "west", "south", "east"]
    exportPreset: Literal["decision_clip", "hooked_flagship"]
    qualificationStatus: Literal["qualified", "provisional", "rejected", "quarantined"]
    auditOnly: Literal[True]
    publicationEligible: Literal[False]
    renderPurpose: Literal["research_audit"]
    qualificationFailures: list[str]
    publicationBlockers: list[str]
    publicationPolicy: ResearchRenderPublicationPolicyResponse
    locator: dict[str, Any]
    bundleArtifact: dict[str, Any]
    basename: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$",
    )
    preview: dict[str, Any]
    source: ResearchRenderSourceProvenanceResponse


class FoundryCampaignRequest(BaseModel):
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    engineVersion: str = Field(min_length=1, max_length=100)
    analysisProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    perspectiveProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selectionProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agentRelease: str = Field(min_length=1, max_length=200)
    seatAgents: dict[str, str]
    lanes: list[Literal["organic", "guided", "disagreement", "historical", "expert"]]
    laneProfileHashes: dict[str, str]
    seedStart: int = Field(ge=0)
    seedCount: int = Field(ge=1, le=1_000_000_000)
    seedsPerShard: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def validate_equal_lineup(self) -> "FoundryCampaignRequest":
        if set(self.seatAgents) != {"north", "west", "south", "east"}:
            raise ValueError("seatAgents must declare all four seats exactly once")
        if set(self.seatAgents.values()) != {self.agentRelease}:
            raise ValueError("all four seats must use the same signed agent release")
        if not self.lanes:
            raise ValueError("at least one discovery lane is required")
        if set(self.laneProfileHashes) != set(self.lanes):
            raise ValueError(
                "laneProfileHashes must declare every selected lane exactly once"
            )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.laneProfileHashes.values()
        ):
            raise ValueError("laneProfileHashes must contain SHA-256 identities")
        return self


class FoundryShardClaimRequest(BaseModel):
    workerId: str = Field(min_length=1, max_length=200)
    campaignHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    supportedLanes: list[
        Literal["organic", "guided", "disagreement", "historical", "expert"]
    ] = Field(default_factory=lambda: ["organic"], min_length=1)
    leaseSeconds: int = Field(int(_FOUNDRY_JOB_POLICY["lease_seconds"]), ge=3, le=3600)


class FoundryShardHeartbeatRequest(BaseModel):
    workerId: str = Field(min_length=1, max_length=200)
    fencingToken: int = Field(ge=1)
    leaseSeconds: int = Field(int(_FOUNDRY_JOB_POLICY["lease_seconds"]), ge=3, le=3600)


class FoundryArtifactReference(BaseModel):
    ownerKind: str = Field(min_length=1, max_length=100)
    ownerHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str = Field(min_length=1, max_length=2000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    mediaType: str = Field(min_length=1, max_length=200)
    name: str | None = Field(None, max_length=500)


class TraceImportProvenanceRequest(BaseModel):
    schemaVersion: Literal[1] = 1
    lane: Literal["historical", "expert"]
    sourceRecordId: str = Field(min_length=1, max_length=500)
    sourceEvidenceHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rightsStatus: Literal["cleared", "research_only"]
    verificationStatus: Literal[
        "verified_action_order", "unreviewed", "verified_expert", "consensus_approved"
    ]
    completeHiddenStateAttested: bool
    submittedByHash: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


class TraceImportCatalogEntryRequest(BaseModel):
    ordinal: int = Field(ge=0)
    traceReference: FoundryArtifactReference
    provenance: TraceImportProvenanceRequest


class TraceImportCatalogCreateRequest(BaseModel):
    lane: Literal["historical", "expert"]
    entries: list[TraceImportCatalogEntryRequest] = Field(
        min_length=1, max_length=10_000
    )


class ScoreReachabilityCatalogCreateRequest(BaseModel):
    traceReferences: list[FoundryArtifactReference] = Field(
        min_length=1, max_length=10_000
    )


class ScoreReachabilityProveRequest(BaseModel):
    targetScore: tuple[int, int]
    targetDealer: Literal["north", "west", "south", "east"]
    initialDealer: Literal["north", "west", "south", "east"] = "east"

    @model_validator(mode="after")
    def validate_target_score(self) -> "ScoreReachabilityProveRequest":
        if min(self.targetScore) < 0:
            raise ValueError("targetScore must contain two nonnegative scores")
        return self


class FoundryDeepAnalysisCreateRequest(BaseModel):
    campaignHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FoundryVideoProductionBatchCreateRequest(BaseModel):
    targetCount: int = Field(20, ge=1, le=100)
    seedStart: int = Field(ge=0)
    profileHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FoundryFinalistValidationCreateRequest(BaseModel):
    analysisJobId: str = Field(min_length=1, max_length=200)
    candidateHashes: list[str] = Field(min_length=1, max_length=100)

    @field_validator("candidateHashes")
    @classmethod
    def validate_candidate_hashes(cls, value: list[str]) -> list[str]:
        normalized = [str(item).strip().lower() for item in value]
        if len(set(normalized)) != len(normalized) or any(
            not re.fullmatch(r"[0-9a-f]{64}", item) for item in normalized
        ):
            raise ValueError("candidate hashes must be unique lowercase SHA-256 values")
        return normalized


class FoundryCampaignProofExecutionRequest(BaseModel):
    campaignHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    declaredWorkerCount: int = Field(ge=1, le=100_000)
    failureSchedule: str = Field(min_length=1, max_length=200)


class FoundryCampaignProofCreateRequest(BaseModel):
    executions: list[FoundryCampaignProofExecutionRequest] = Field(
        min_length=2, max_length=100
    )


class FoundryCatalogRecordResponse(BaseModel):
    catalogHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal[
        "trace_import_catalog_v1",
        "score_reachability_catalog_v1",
        "score_reachability_proof_v1",
        "campaign_equivalence_proof_v1",
        "context_search_result_v1",
        "natural_sakka_generation_v1",
        "natural_sakka_v1",
        "curated_sakka_v1",
        "qualified_laffa_v1",
        "qualified_laffa_v2",
    ]
    artifact: dict[str, Any]
    metadata: dict[str, Any]
    createdAt: str


class FoundryCatalogDocumentResponse(BaseModel):
    record: FoundryCatalogRecordResponse
    document: dict[str, Any]


class FoundryCatalogListResponse(BaseModel):
    items: list[FoundryCatalogRecordResponse]
    nextCursor: str | None


class FoundryShardCommitRequest(BaseModel):
    workerId: str = Field(min_length=1, max_length=200)
    fencingToken: int = Field(ge=1)
    processedSeeds: list[int]
    traceHashes: list[str]
    candidateHashes: list[str]
    coverageCells: list[str]
    artifactReferences: list[FoundryArtifactReference] = Field(default_factory=list)
    completionStatus: Literal["completed"] = "completed"


class FoundryChallengeLocatorCreateRequest(BaseModel):
    laffaIdentifier: str = Field(min_length=1, max_length=200)
    decisionId: str = Field(min_length=1, max_length=200)
    observerSeat: Literal["north", "west", "south", "east"]
    promptContract: Literal[
        "choose_acting_action",
        "predict_opponent_play",
        "read_partner_signal",
        "identify_threat",
        "plan_next_response",
        "infer_distribution",
        "post_reveal_analysis",
    ]
    revealPolicy: Literal["after_commit", "after_completion", "post_reveal_only"] = (
        "after_commit"
    )
    expiresAt: int | None = Field(None, ge=1)
    analysisArtifact: FoundryArtifactReference | None = None


class FoundryChallengeAttemptRequest(BaseModel):
    participantId: str | None = Field(None, min_length=1, max_length=200)
    chosenAction: dict[str, Any]
    responseSeconds: float | None = Field(None, ge=0.0, le=3600.0)
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    changedAnswer: bool = False


class FoundryCampaignResponse(BaseModel):
    campaignHash: str
    spec: dict[str, Any]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    cancelRequested: bool
    shardCounts: dict[str, int]
    createdAt: str
    updatedAt: str


class FoundryCampaignListResponse(BaseModel):
    items: list[FoundryCampaignResponse]
    nextCursor: str | None


class FoundryRuntimeProfileResponse(BaseModel):
    schemaVersion: Literal[1]
    runtimeBundleRevisionId: str
    runtimeBundleVersion: str
    rulesHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundleHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    engineVersion: str
    analysisProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    searchBackend: dict[str, Any]
    matchValueProfile: dict[str, Any]
    perspectiveProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selectionProfileHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agentRelease: str
    seatAgents: dict[str, str]
    staticLaneProfileHashes: dict[str, str]
    laneAgentReleases: dict[str, str]
    laneSeatAgents: dict[str, dict[str, str]]
    organicScreeningProfile: dict[str, Any]
    guidedSearchProfile: dict[str, Any]
    constraintDealGenerator: dict[str, Any]
    candidateResimulationProfile: dict[str, Any]
    candidateResimulationAgentRelease: str
    finalistValidationProfile: dict[str, Any]
    finalistAgentRelease: str
    finalistSeatAgents: dict[str, str]
    finalistAgentQualification: dict[str, Any]
    finalistAgentPilotEvidence: dict[str, Any]
    catalogRequiredLanes: list[str]
    deepAnalysisProfile: dict[str, Any]
    decisionAnalysisExecutorProfile: dict[str, Any]
    campaignProofProfile: dict[str, Any]
    sakkaGenerationProfile: dict[str, Any]
    videoProductionBatchProfile: dict[str, Any]


class EngineGuideFactResponse(BaseModel):
    label: str
    value: str


class EngineGuideCardResponse(BaseModel):
    rank: str
    points: int
    asset: str


class EngineGuideCardTableResponse(BaseModel):
    title: str
    cards: list[EngineGuideCardResponse]


class EngineGuideSectionResponse(BaseModel):
    id: str
    title: str
    summary: str
    facts: list[EngineGuideFactResponse]
    examples: list[str]
    tables: list[EngineGuideCardTableResponse]


class EngineGuideProfileResponse(BaseModel):
    label: str
    id: str
    version: str
    status: str
    expertRatified: bool


class EngineGuideProofBoundaryResponse(BaseModel):
    title: str
    body: str


class EngineGuideResponse(BaseModel):
    schemaVersion: Literal[1]
    locale: Literal["ar-SA"]
    direction: Literal["rtl"]
    pageTitle: str
    brand: str
    subtitle: str
    interface: dict[str, str]
    profile: EngineGuideProfileResponse
    proofBoundary: EngineGuideProofBoundaryResponse
    sections: list[EngineGuideSectionResponse]
    capabilityReport: dict[str, Any]
    ruleSourceCatalog: dict[str, Any] | None = None
    adjudicationAuthority: dict[str, Any] | None = None
    audit: dict[str, str]
    guideHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class FoundryShardResponse(BaseModel):
    specHash: str
    campaignHash: str
    shardIndex: int
    spec: dict[str, Any]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    workerId: str | None = None
    leaseUntil: float | None = None
    fencingToken: int
    attempts: int
    result: dict[str, Any] | None = None
    error: str | None = None


class FoundryShardListResponse(BaseModel):
    campaignHash: str
    shards: list[FoundryShardResponse]


class FoundryShardAttemptResponse(BaseModel):
    specHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaignHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    shardIndex: int = Field(ge=0)
    fencingToken: int = Field(ge=1)
    attempt: int = Field(ge=1)
    workerId: str
    claimedAt: float
    finishedAt: float | None = None
    terminalStatus: str | None = None
    error: str | None = None


class FoundryShardAttemptListResponse(BaseModel):
    campaignHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: list[FoundryShardAttemptResponse]


class FoundryShardClaimResponse(BaseModel):
    claim: dict[str, Any] | None


class FoundryDeepAnalysisJobResponse(BaseModel):
    jobId: str
    tenantId: str
    kind: Literal["foundry_deep_analysis_v1"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    priority: int
    progress: float
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int
    maxAttempts: int
    leaseOwner: str | None = None
    leaseUntil: float | None = None
    cancelRequested: bool
    createdAt: str
    updatedAt: str


class FoundryDeepAnalysisJobListResponse(BaseModel):
    items: list[FoundryDeepAnalysisJobResponse]


class FoundryVideoProductionBatchJobResponse(BaseModel):
    jobId: str
    tenantId: str
    kind: Literal["foundry_video_batch_v1"]
    status: Literal[
        "queued", "running", "completed", "failed", "cancelled", "exhausted"
    ]
    priority: int
    progress: float
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int
    maxAttempts: int
    leaseOwner: str | None = None
    leaseUntil: float | None = None
    cancelRequested: bool
    createdAt: str
    updatedAt: str


class FoundryVideoProductionBatchJobListResponse(BaseModel):
    items: list[FoundryVideoProductionBatchJobResponse]


class QualifiedLaffaLibraryResponse(BaseModel):
    items: list[dict[str, Any]]
    nextCursor: str | None = None


class FoundryFinalistValidationJobResponse(BaseModel):
    jobId: str
    tenantId: str
    kind: Literal["foundry_finalist_validation_v1"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    priority: int
    progress: float
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int
    maxAttempts: int
    leaseOwner: str | None = None
    leaseUntil: float | None = None
    cancelRequested: bool
    createdAt: str
    updatedAt: str


class FoundryFinalistValidationJobListResponse(BaseModel):
    items: list[FoundryFinalistValidationJobResponse]


class FoundryAnalysisCandidateListResponse(BaseModel):
    schemaVersion: Literal[1]
    jobId: str
    campaignHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    count: int = Field(ge=0)
    items: list[dict[str, Any]]


class FoundryAnalysisCandidatePrepareRequest(BaseModel):
    revealPolicy: Literal["after_commit", "after_completion", "post_reveal_only"] = (
        "after_commit"
    )
    exportPreset: Literal["decision_clip", "hooked_flagship"] = "decision_clip"


class FoundryAnalysisCandidatePrepareResponse(BaseModel):
    schemaVersion: Literal[1]
    exportPreset: Literal["decision_clip", "hooked_flagship"]
    candidate: dict[str, Any]
    locator: dict[str, Any]
    bundleArtifact: FoundryArtifactReference
    basename: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$",
    )


class FoundryRenderSourceReference(ResearchRenderSourceProvenanceResponse):
    pass


class FoundryRenderBundleArtifactReference(FoundryArtifactReference):
    model_config = {"extra": "forbid"}


class FoundryRenderProofArtifactReference(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    uri: str = Field(min_length=1, max_length=2000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)
    mediaType: Literal[
        "application/vnd.baloot.research-proof-outcome-v1+json",
        "application/vnd.balootx.ortools-proof-outcome-v1+json",
    ]
    name: str = Field(min_length=1, max_length=500)


class FoundryRenderPlayerLocator(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

    schemaVersion: Literal[1]
    code: str = Field(pattern=r"^LAF-[A-F0-9]{10}$")
    universalLink: str = Field(pattern=r"^/c/LAF-[A-F0-9]{10}$")

    @model_validator(mode="after")
    def validate_result_binding(self) -> "FoundryRenderPlayerLocator":
        if self.universalLink != f"/c/{self.code}":
            raise ValueError("render player locator must match its result code")
        return self


class FoundryRenderBatchItemRequest(BaseModel):
    model_config = {"extra": "forbid"}

    bundleArtifact: FoundryRenderBundleArtifactReference
    basename: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$",
    )
    source: FoundryRenderSourceReference | None = None
    proofArtifact: FoundryRenderProofArtifactReference | None = None
    authorityReceipt: ResearchScenarioAuthorityReceiptResponse | None = None
    locator: FoundryRenderPlayerLocator | None = None

    @model_validator(mode="after")
    def validate_player_locator_authority(self) -> "FoundryRenderBatchItemRequest":
        if self.locator is None:
            return self
        source = self.source
        authority = self.authorityReceipt
        if (
            source is None
            or source.kind != "research_scenario_result"
            or source.code != self.locator.code
            or source.auditOnly is not False
            or source.publicationEligible is not True
            or source.renderPurpose != "content_candidate"
            or authority is None
            or authority.resultCode != self.locator.code
        ):
            raise ValueError(
                "render player locator requires its qualified prepared result authority"
            )
        return self


class FoundryRenderBatchCreateRequest(BaseModel):
    model_config = {"extra": "forbid"}

    items: list[FoundryRenderBatchItemRequest] = Field(min_length=1, max_length=100)
    playbackEvidence: StrictBool = False

    @model_validator(mode="after")
    def validate_basenames(self) -> "FoundryRenderBatchCreateRequest":
        basenames = [item.basename for item in self.items]
        if len(set(basenames)) != len(basenames):
            raise ValueError("render batch basenames must be unique")
        return self


class FoundryRenderBatchJobResponse(BaseModel):
    jobId: str
    tenantId: str
    kind: Literal["render_batch_v1"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    priority: int
    progress: float
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int
    maxAttempts: int
    leaseOwner: str | None = None
    leaseUntil: float | None = None
    cancelRequested: bool
    createdAt: str
    updatedAt: str


class FoundryRenderBatchJobListResponse(BaseModel):
    items: list[FoundryRenderBatchJobResponse]


class ResearchResultRenderJobResponse(FoundryRenderBatchJobResponse):
    model_config = {"extra": "forbid"}

    result: dict[str, Any] | None
    error: str | None
    leaseOwner: str | None
    leaseUntil: float | None


class ResearchResultRenderListResponse(BaseModel):
    model_config = {"extra": "forbid"}

    resultCode: str = Field(pattern=r"^LAF-[A-Z0-9]{10}$")
    items: list[ResearchResultRenderJobResponse]


class ChallengeStatusRequest(BaseModel):
    status: Literal["candidate", "review", "rejected"]


class ChallengePublicationReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]
    evidenceSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comment: str = Field(default="", max_length=2000)


class ChallengeGovernanceRequest(BaseModel):
    evidenceSha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChallengeRenderRequest(BaseModel):
    rendererPlugin: str = "auto"
    theme: Literal["emerald", "noir", "royal"] = "emerald"
    width: Literal[720, 1080] = 1080
    height: Literal[1280, 1920] = 1920
    fps: Literal[24, 30, 60] = 60
    crf: int = Field(18, ge=10, le=35)

    @model_validator(mode="after")
    def validate_mobile_canvas(self) -> "ChallengeRenderRequest":
        if self.width * 16 != self.height * 9:
            raise ValueError(
                "release video resolution must use an exact 9:16 aspect ratio"
            )
        return self


class HistoricalReplayRenderRequest(ChallengeRenderRequest):
    decisionId: str | None = None
    rerunAudit: bool = False
    auditSamples: int = Field(48, ge=2, le=1000)
    maximumAuditDecisions: int = Field(40, ge=1, le=128)


def _resolved_hash(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bundle_for_mode(mode: str):
    return ACTIVE_BUNDLE if mode == "full" else build_bundle(ROOT, mode=mode)


def _rules_for(req: VideoJobRequest) -> tuple[RuleBook, dict, dict, str]:
    """Resolve the immutable effective bundle for a request.

    Every user-facing override is folded into the returned content hash.  The
    renderer, analysis and engine therefore cannot report the base profile hash
    after semantically changing a rule or video setting.
    """
    bundle = _bundle_for_mode(req.rulesProfile)
    data = copy.deepcopy(bundle.rules.data)
    data["play"]["first_leader"] = req.firstLeaderPolicy
    data["play"]["configured_first_leader"] = req.configuredFirstLeader
    data["play"]["must_trump_when_void"] = req.mustTrumpWhenVoid
    data["play"]["must_overtrump_when_possible"] = req.mustOvertrumpWhenPossible
    data["play"]["allow_discard_when_cannot_overtrump"] = (
        req.allowDiscardWhenCannotOvertrump
    )
    data["play"]["partner_winning_exception"] = req.partnerWinningException
    data["projects"]["enabled"] = req.projectsEnabled
    data["projects"]["baloot"]["enabled"] = req.balootEnabled
    data["scoring"]["kabout"]["enabled"] = req.kaboutEnabled
    data["scoring"]["match"]["target"] = req.matchTarget
    if not req.ashkalEnabled:
        data["contracts"]["allowed"] = [
            value for value in data["contracts"]["allowed"] if value != "ashkal"
        ]
        data["bidding"]["first_round"]["allowed"] = [
            value
            for value in data["bidding"]["first_round"]["allowed"]
            if value != "ashkal"
        ]
        data["bidding"]["ashkal"]["enabled"] = False
    if req.cutMode == "none":
        data["dealing"]["cut_enabled"] = False
    rules = RuleBook(data)

    convention = copy.deepcopy(bundle.convention)
    convention["display_observations"] = req.conventionProfile in {
        "experimental_display",
        "experimental_weighting",
    }
    convention["apply_to_probability"] = (
        req.conventionProfile == "experimental_weighting"
    )

    video = copy.deepcopy(bundle.video)
    themes = video.get("themes", {})
    if req.theme in themes:
        video["colors"] = copy.deepcopy(themes[req.theme])
    video["selected_theme"] = req.theme
    video["renderer_plugin"] = req.rendererPlugin
    video["render"]["width"] = req.width
    video["render"]["height"] = req.height
    video["render"]["fps"] = req.fps
    video["render"]["crf"] = req.crf
    video.setdefault("layout", {})["orient_side_cards_to_table"] = (
        req.cardOrientation == "table"
    )
    # Pacing scales presentation-only durations. The viewer decision pause is
    # always set explicitly and is therefore not shortened accidentally.
    pacing_factor = {
        "fast": 0.58,
        "broadcast": 0.78,
        "authentic": 1.0,
        "cinematic": 1.18,
    }[req.pacing]
    video["selected_pacing"] = req.pacing
    for key, value in list(video.get("timeline", {}).items()):
        if key not in {"pause_seconds", "transition_seconds"} and isinstance(
            value, (int, float)
        ):
            video["timeline"][key] = round(float(value) * pacing_factor, 4)
    video["timeline"]["pause_seconds"] = req.pauseSeconds
    video.setdefault("callouts", {})["enabled"] = req.seatCallouts
    video["callouts"]["seat_attached"] = req.seatCallouts
    video.setdefault("presentation", {})["show_public_callouts"] = req.seatCallouts
    video.setdefault("audio", {})["enabled"] = req.audio
    video["audio"]["voice_callouts"] = req.voiceCallouts
    video["audio"]["table_foley"] = req.tableFoley
    video["audio"]["spoken_voice_pack"] = req.spokenVoicePack
    video["card_assets"] = CARD_PACKS.profile_fragment(req.cardPack, req.cardBack)

    identity = {
        "baseBundle": bundle.sha256,
        "rules": rules.sha256,
        "conventions": convention,
        "video": video,
    }
    return rules, convention, video, _resolved_hash(identity)


_LEGACY_VIDEO_V4_SUPPORTED_FIELDS = frozenset(
    {
        "bundleRevisionId",
        "scope",
        "seed",
        "rulesProfile",
        "dealer",
        "contract",
        "trumpSuit",
        "buyer",
        "matchScoreUs",
        "matchScoreThem",
        "escalationLevel",
        "escalationMode",
        "requireProject",
        "pauseTrickMin",
        "pauseTrickMax",
        "maxSeedAttempts",
        "maxCandidatesPerSeed",
    }
)
_LEGACY_VIDEO_V4_SUCCESSORS = {
    "author": "/api/v2/research-recipes",
    "run": "/api/v2/research-runs",
    "results": "/api/v2/research-results?runCode={runCode}",
    "prepareRender": "/api/v2/research-results/{resultCode}:prepare-render",
    "render": "/api/v2/render-batches",
}
_LEGACY_VIDEO_V4_OPENAPI = {
    "x-baloot-authority": "research_v4",
    "x-baloot-successor-endpoints": list(_LEGACY_VIDEO_V4_SUCCESSORS.values()),
}
_LEGACY_VIDEO_V4_CREATE_OPENAPI = {
    **_LEGACY_VIDEO_V4_OPENAPI,
    "parameters": [
        {
            "name": "Idempotency-Key",
            "in": "header",
            "required": True,
            "schema": {
                "type": "string",
                "minLength": 1,
                "maxLength": IDEMPOTENCY_KEY_MAX_LENGTH,
                "pattern": IDEMPOTENCY_KEY_PATTERN,
            },
        }
    ],
}


class LegacyVideoCompatibilityError(ValueError):
    """A direct video request cannot be represented without changing meaning."""

    def __init__(self, code: str, *, fields: list[str] | None = None):
        core = {
            "schemaVersion": 1,
            "code": code,
            "authority": "research_v4",
            "unsupportedFields": sorted(fields or []),
            "successorEndpoints": copy.deepcopy(_LEGACY_VIDEO_V4_SUCCESSORS),
        }
        self.diagnostic = {**core, "sha256": _resolved_hash(core)}
        super().__init__(code)


def _legacy_video_predicate(
    predicate_id: str,
    kind: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": copy.deepcopy(arguments),
    }


def _legacy_video_v4_recipe(request: VideoJobRequest) -> dict[str, Any]:
    """Compile the representable legacy subset into the sole scenario authority."""

    if not request.bundleRevisionId:
        raise LegacyVideoCompatibilityError(
            "bundle_revision_required",
            fields=["bundleRevisionId"],
        )
    if request.scope != "round":
        raise LegacyVideoCompatibilityError(
            "legacy_match_generation_unsupported",
            fields=["scope", "maximumRounds"],
        )
    if request.rulesProfile != "full":
        raise LegacyVideoCompatibilityError(
            "legacy_rule_variant_unsupported",
            fields=["rulesProfile"],
        )
    if request.escalationLevel != "normal" or request.escalationMode != "open":
        raise LegacyVideoCompatibilityError(
            "legacy_play_escalation_unsupported",
            fields=[
                field
                for field, value, default in (
                    ("escalationLevel", request.escalationLevel, "normal"),
                    ("escalationMode", request.escalationMode, "open"),
                )
                if value != default
            ],
        )
    if request.maxCandidatesPerSeed != 1:
        raise LegacyVideoCompatibilityError(
            "legacy_video_candidates_per_seed_unsupported",
            fields=["maxCandidatesPerSeed"],
        )

    unsupported = sorted(
        field
        for field in request.model_fields_set
        if field not in _LEGACY_VIDEO_V4_SUPPORTED_FIELDS
    )
    if unsupported:
        raise LegacyVideoCompatibilityError(
            "legacy_video_overrides_unsupported",
            fields=unsupported,
        )

    # The exact proof tier admits at most sixteen remaining cards.  Select one
    # deterministic late-game trick from the requested legacy range and fail
    # before persistence when that range cannot be represented truthfully.
    target_trick = min(request.pauseTrickMax, 6)
    if target_trick < max(request.pauseTrickMin, 4):
        raise LegacyVideoCompatibilityError(
            "legacy_video_trick_range_unsupported",
            fields=["pauseTrickMin", "pauseTrickMax"],
        )

    constraints: list[dict[str, Any]] = [
        _legacy_video_predicate(
            "PRED-LEGACY-DEALER",
            "auction.dealer",
            {"seat": request.dealer},
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-ESCALATION",
            "escalation.level",
            {"level": request.escalationLevel},
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-ESCALATION-MODE",
            "escalation.locked",
            {"locked": request.escalationMode == "locked"},
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-MATCH-US",
            "state.match_score",
            {
                "team": "north_south",
                "minimum": request.matchScoreUs,
                "maximum": request.matchScoreUs,
            },
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-MATCH-THEM",
            "state.match_score",
            {
                "team": "west_east",
                "minimum": request.matchScoreThem,
                "maximum": request.matchScoreThem,
            },
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-TRICK",
            "play.trick_index",
            {
                "minimum": target_trick,
                "maximum": target_trick,
            },
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-ACTOR",
            "play.current_player",
            {"seat": "south"},
        ),
        _legacy_video_predicate(
            "PRED-LEGACY-LEGAL-ACTIONS",
            "state.legal_action_count",
            {"minimum": 2, "maximum": 8},
        ),
    ]
    if request.contract != "auto":
        constraints.append(
            _legacy_video_predicate(
                "PRED-LEGACY-CONTRACT",
                "auction.contract",
                {"contract": request.contract},
            )
        )
    if request.trumpSuit != "auto":
        constraints.append(
            _legacy_video_predicate(
                "PRED-LEGACY-TRUMP",
                "auction.trump",
                {"suit": request.trumpSuit},
            )
        )
    if request.buyer != "auto":
        constraints.append(
            _legacy_video_predicate(
                "PRED-LEGACY-BUYER",
                "auction.buyer",
                {"seat": request.buyer},
            )
        )
    if request.requireProject:
        constraints.append(
            {
                "op": "any",
                "children": [
                    _legacy_video_predicate(
                        "PRED-LEGACY-PROJECT-US",
                        "project.count",
                        {
                            "team": "north_south",
                            "minimum": 1,
                            "maximum": 8,
                            "state": "eligible",
                        },
                    ),
                    _legacy_video_predicate(
                        "PRED-LEGACY-PROJECT-THEM",
                        "project.count",
                        {
                            "team": "west_east",
                            "minimum": 1,
                            "maximum": 8,
                            "state": "eligible",
                        },
                    ),
                ],
            }
        )

    seeds_per_shard = next(
        candidate
        for candidate in range(min(4, request.maxSeedAttempts), 0, -1)
        if request.maxSeedAttempts % candidate == 0
    )
    return {
        "schemaVersion": 4,
        "titleAr": _VIDEO_JOB_TITLE_DEFAULT,
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": request.bundleRevisionId,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": constraints},
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    # Preserve a non-empty partial trick while leaving the
                    # acting seat at least two cards for a real decision.
                    "remainingCards": 31 - (4 * target_trick),
                    "actingSeat": "south",
                },
            },
            "outcome": _legacy_video_predicate(
                "PRED-LEGACY-ROOT-COUNT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": 8},
            ),
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "winning_root_action_count",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "late_game_exact_v1",
        },
        "qualification": {},
        "resources": {
            "generationLane": "ortools",
            "generationPolicy": "ortools_cp_sat_v3",
            "seedStart": request.seed,
            "seedCount": request.maxSeedAttempts,
            "seedsPerShard": seeds_per_shard,
            "solverTimeoutSeconds": 15.0,
            "proofBudgetProfile": "late_game_exact_v1",
        },
        "diversity": {
            "schemaVersion": 1,
            "cellDimensions": [
                "contract",
                "project",
                "decision_kind",
                "proof_mode",
            ],
            "topKPerCell": 3,
            "rejectedAuditModulus": 100,
        },
        "renderPolicy": {"mode": "on_demand"},
        "status": "published",
    }


def _submit_legacy_video_compatibility(
    request: VideoJobRequest,
    *,
    idempotency_key: str,
) -> dict[str, Any]:
    recipe_payload = _legacy_video_v4_recipe(request)
    validated_key = validate_idempotency_key(idempotency_key)
    key_material = hashlib.sha256(validated_key.encode("utf-8")).hexdigest()[:32]
    recipe = RESEARCH.create_recipe(
        recipe_payload,
        idempotency_key=f"legacy-video-recipe-{key_material}",
    )
    run = RESEARCH.submit(
        str(recipe["code"]),
        idempotency_key=f"legacy-video-run-{key_material}",
    )
    return {
        "jobId": run["runCode"],
        "status": run["status"],
        "progress": float(run.get("progress", 0.0)),
        "authority": "research_v4",
        "recipeCode": recipe["code"],
        "bundleRevisionId": request.bundleRevisionId,
        "rendersAutomatically": False,
        "deprecation": {
            "schemaVersion": 1,
            "status": "deprecated",
            "successorEndpoints": copy.deepcopy(_LEGACY_VIDEO_V4_SUCCESSORS),
        },
    }


def _legacy_video_run_is_v4(run: dict[str, Any]) -> bool:
    return bool(str(run.get("scenarioExecutorIdentityHash") or ""))


def _legacy_video_run_view(run: dict[str, Any]) -> dict[str, Any]:
    if not _legacy_video_run_is_v4(run):
        raise ValueError("legacy video adapter accepts only Research V4 runs")
    return {
        "jobId": run["runCode"],
        "status": run["status"],
        "progress": float(run.get("progress", 0.0)),
        "kind": "research_v4_compatibility",
        "authority": "research_v4",
        "recipeCode": run["recipeCode"],
        "resultCount": int(run.get("resultCount", 0)),
        "qualifiedCount": int(run.get("qualifiedCount", 0)),
        "quarantineCount": int(run.get("quarantineCount", 0)),
        "rendersAutomatically": False,
        "deprecation": {
            "schemaVersion": 1,
            "status": "deprecated",
            "successorEndpoints": copy.deepcopy(_LEGACY_VIDEO_V4_SUCCESSORS),
        },
    }


def _parse_contract(value: str) -> Contract | None:
    return None if value == "auto" else Contract.parse(value)


def _parse_suit(value: str) -> Suit | None:
    return None if value == "auto" else Suit.parse(value)


def _parse_seat(value: str) -> Seat | None:
    return None if value == "auto" else Seat.parse(value)


def _ensure_not_cancelled(job_id: str, store: Any = None) -> None:
    store = store or STORE
    if store.cancelled(job_id):
        store.update(
            job_id, status="cancelled", progress=1.0, error="cancelled by request"
        )
        raise InterruptedError("job cancelled")


def _record_render_artifacts(
    artifacts: Any, episode: Any, *, owner_kind: str, owner_id: str
) -> list[dict[str, Any]]:
    """Persist one video and all decision-to-timecode relationships.

    The video plugin writes an immutable manifest. This adapter only indexes
    that manifest; it never derives or changes Baloot semantics.
    """
    manifest = json.loads(Path(artifacts.manifest).read_text(encoding="utf-8"))
    episode_bytes = Path(artifacts.episode_json).read_bytes()
    episode_hash = hashlib.sha256(episode_bytes).hexdigest()
    game_hash = getattr(episode, "game_record_hash", None) or manifest.get(
        "gameRecordHash"
    )
    renderer_id = str((manifest.get("renderer") or {}).get("id", "unknown"))
    if getattr(artifacts, "timeline", None) is not None:
        timeline_payload = json.loads(
            Path(artifacts.timeline).read_text(encoding="utf-8")
        )
    else:
        timeline_payload = {
            "schemaVersion": 1,
            "segments": list(manifest.get("timelineIndex") or []),
        }
    timeline = list(timeline_payload.get("segments") or [])
    common = {
        "gameHash": game_hash,
        "episodeHash": episode_hash,
        "rendererId": renderer_id,
        "manifest": {
            **manifest,
            "urls": {
                "video": f"/outputs/{Path(artifacts.video).name}",
                "poster": f"/outputs/{Path(artifacts.poster).name}",
                "episode": f"/outputs/{Path(artifacts.episode_json).name}",
                "captions": f"/outputs/{Path(artifacts.captions).name}",
                "captionsVtt": (
                    f"/outputs/{Path(artifacts.captions_vtt).name}"
                    if getattr(artifacts, "captions_vtt", None)
                    else None
                ),
                "transcript": (
                    f"/outputs/{Path(artifacts.transcript).name}"
                    if getattr(artifacts, "transcript", None)
                    else None
                ),
                "contactSheet": f"/outputs/{Path(artifacts.contact_sheet).name}",
                "manifest": f"/outputs/{Path(artifacts.manifest).name}",
                "timeline": f"/outputs/{Path(artifacts.timeline).name}"
                if getattr(artifacts, "timeline", None)
                else None,
                "semanticScenes": [
                    f"/outputs/{Path(path).name}"
                    for path in getattr(artifacts, "semantic_files", ())
                ],
            },
        },
        "timeline": timeline_payload,
    }
    records: list[dict[str, Any]] = []
    targets: list[tuple[str, str]] = [(owner_kind, str(owner_id))]
    if game_hash:
        targets.append(("game", str(game_hash)))
    decision_ids = {
        str(item.get("decisionId"))
        for item in timeline + list(manifest.get("decisionIndex") or [])
        if item.get("decisionId")
    }
    targets.extend(("decision", decision_id) for decision_id in sorted(decision_ids))
    seen: set[tuple[str, str]] = set()
    for target_kind, target_id in targets:
        if (target_kind, target_id) in seen:
            continue
        seen.add((target_kind, target_id))
        records.append(
            LEARNING_STORE.put_video_record(
                {**common, "ownerKind": target_kind, "ownerId": target_id}
            )
        )
    return records


def _execute_job(
    job_id: str,
    req_dict: dict[str, Any],
    *,
    store: Any = None,
    outputs: Path | None = None,
    registry: PluginRegistry | None = None,
) -> None:
    store = store or STORE
    outputs = outputs or OUTPUTS
    registry = registry or REGISTRY
    try:
        req = VideoJobRequest.model_validate(req_dict)
        _ensure_not_cancelled(job_id, store)
        store.update(job_id, status="running", progress=0.05)
        rules, convention, video_profile, bundle_hash = _rules_for(req)
        contract = _parse_contract(req.contract)
        if req.rulesProfile == "sun_only":
            contract = Contract.SUN
        if req.rulesProfile == "hokum_only":
            contract = Contract.HOKUM
        cut_position = req.cutPosition if req.cutMode == "fixed" else None
        scenario = ScenarioRequest.governed(
            seed=req.seed,
            viewer=Seat.SOUTH,
            dealer=Seat.parse(req.dealer),
            cut_position=cut_position,
            contract=contract,
            buyer=_parse_seat(req.buyer),
            trump_suit=_parse_suit(req.trumpSuit),
            match_score=(req.matchScoreUs, req.matchScoreThem),
            escalation_level=req.escalationLevel,
            escalation_mode=req.escalationMode,
            require_project=req.requireProject,
            pause_trick_min=req.pauseTrickMin,
            pause_trick_max=req.pauseTrickMax,
            requested_samples=req.samples,
            exact_allocation_limit=req.exactAllocationLimit,
            exact_search_max_remaining=req.exactSearchRemaining,
            max_seed_attempts=req.maxSeedAttempts,
            max_candidates_per_seed=req.maxCandidatesPerSeed,
            minimum_hidden_deals=req.minimumHiddenDeals,
            minimum_expected_margin_gap=req.minimumExpectedMarginGap,
            minimum_win_probability_gap=req.minimumWinProbabilityGap,
            require_exact_analysis=req.requireExactAnalysis,
            target_debate_score=req.targetDebateScore,
        )
        store.update(job_id, progress=0.20)
        _ensure_not_cancelled(job_id, store)
        if req.scope == "match":
            match = generate_match_episode(
                rules,
                convention,
                bundle_hash=bundle_hash,
                copy=video_profile["localizations"],
                request=MatchScenarioRequest.governed(
                    seed=req.seed,
                    dealer=Seat.parse(req.dealer),
                    viewer=Seat.SOUTH,
                    maximum_rounds=req.maximumRounds,
                    contract=contract,
                    trump_suit=_parse_suit(req.trumpSuit),
                    require_project=req.requireProject,
                    requested_samples=req.samples,
                    exact_search_max_remaining=req.exactSearchRemaining,
                ),
            )
            store.update(job_id, progress=0.45)
            _ensure_not_cancelled(job_id, store)
            plugin = registry.get("video_renderer", req.rendererPlugin)
            if "multi_round_match" not in plugin.descriptor.capabilities:
                raise ValueError(
                    f"renderer {req.rendererPlugin} does not support match scope"
                )
            artifacts = plugin.execute(
                match,
                output_dir=outputs,
                basename=f"baloot-{job_id}",
                video_profile=video_profile,
            )
            video_records = _record_render_artifacts(
                artifacts, match, owner_kind="match", owner_id=match.match_id
            )
            item_id = match.match_id
            round_count = len(match.rounds)
        else:
            episode = generate_episode(
                rules,
                convention,
                bundle_hash=bundle_hash,
                request=scenario,
                copy=video_profile["localizations"],
            )
            episode = replace(episode, title_ar=req.titleAr, question_ar=req.questionAr)
            store.update(job_id, progress=0.45)
            _ensure_not_cancelled(job_id, store)
            plugin = registry.get("video_renderer", req.rendererPlugin)
            if "multi_round_match" in plugin.descriptor.capabilities:
                raise ValueError(f"renderer {req.rendererPlugin} requires match scope")
            artifacts = plugin.execute(
                episode,
                export_preset="hooked_flagship",
                output_dir=outputs,
                basename=f"baloot-{job_id}",
                video_profile=video_profile,
            )
            video_records = _record_render_artifacts(
                artifacts, episode, owner_kind="episode", owner_id=episode.episode_id
            )
            item_id = episode.episode_id
            round_count = 1
        _ensure_not_cancelled(job_id, store)
        result = {
            "video": f"/outputs/{artifacts.video.name}",
            "poster": f"/outputs/{artifacts.poster.name}",
            "episode_json": f"/outputs/{artifacts.episode_json.name}",
            "captions": f"/outputs/{artifacts.captions.name}",
            "captions_vtt": f"/outputs/{artifacts.captions_vtt.name}"
            if artifacts.captions_vtt
            else None,
            "transcript": f"/outputs/{artifacts.transcript.name}"
            if artifacts.transcript
            else None,
            "contact_sheet": f"/outputs/{artifacts.contact_sheet.name}",
            "manifest": f"/outputs/{artifacts.manifest.name}",
            "timeline": f"/outputs/{artifacts.timeline.name}"
            if artifacts.timeline
            else None,
            "itemId": item_id,
            "scope": req.scope,
            "roundCount": round_count,
            "bundleHash": bundle_hash,
            "videoRecords": [item.get("videoId") for item in video_records],
        }
        store.update(job_id, status="completed", progress=1.0, result=result)
    except InterruptedError:
        if not store.cancelled(job_id):
            store.update(job_id, status="cancelled", progress=1.0, error="cancelled")
    except Exception as exc:
        store.update(
            job_id,
            status="failed",
            progress=1.0,
            error=f"{exc}\n{traceback.format_exc(limit=5)}",
        )


def _execute_challenge_job(job_id: str, request_data: dict[str, Any]) -> None:
    """Execute one logical mining request across requested source/kind lanes.

    ``both`` and ``any`` are expanded here rather than hidden inside the miner,
    keeping each miner invocation deterministic and independently diagnosable.
    Results are globally ranked and deduplicated before returning.
    """
    try:
        request = ChallengeMiningRequest.model_validate(request_data)
        STORE.update(job_id, status="running", progress=0.03)
        miner = REGISTRY.get("challenge_miner", "progressive_challenge_miner_v2")
        sources = (
            ("simulation", "history") if request.source == "both" else (request.source,)
        )
        kinds = (
            ("play", "bid")
            if request.decisionKind == "any"
            else (request.decisionKind,)
        )
        lanes = [(source, kind) for source in sources for kind in kinds]
        combined: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        # Divide only result quota; each lane keeps the configured statistical
        # budgets so comparisons remain meaningful. A common deadline is split
        # evenly to prevent one sparse lane from starving the others.
        lane_results = max(1, math.ceil(request.maximumResults / max(1, len(lanes))))
        lane_seconds = max(1.0, request.maximumSeconds / max(1, len(lanes)))
        for index, (source, kind) in enumerate(lanes):
            STORE.update(
                job_id,
                status="running",
                progress=0.05 + 0.78 * index / max(1, len(lanes)),
            )
            lane = MiningRequest(
                source=source,
                decision_kind=kind,
                seed_start=request.seedStart + index * max(1, request.seeds),
                seeds=request.seeds,
                contract=request.contract,
                buyer_relationship=request.buyerRelationship,
                archetype=request.archetype,
                difficulty_min=request.difficultyMin,
                difficulty_max=request.difficultyMax,
                minimum_publishability=request.minimumPublishability,
                maximum_results=lane_results,
                require_project=request.requireProject,
                pause_trick_min=request.pauseTrickMin,
                pause_trick_max=request.pauseTrickMax,
                maximum_seconds=lane_seconds,
                player_profiles=tuple(
                    sorted((str(k), str(v)) for k, v in request.playerProfiles.items())
                ),
                shallow_samples=request.shallowSamples,
                deep_samples=request.deepSamples,
                exact_allocation_limit=request.exactAllocationLimit,
                exact_search_remaining=request.exactSearchRemaining,
                content_contract=request.contentContract,
            )
            try:
                items = miner.mine(lane)
                combined.extend(items)
                diagnostics.append(
                    {
                        "source": source,
                        "decisionKind": kind,
                        **dict(getattr(miner, "last_run", {})),
                    }
                )
            except Exception as lane_exc:
                # A sparse history lane must not destroy valid simulation results.
                diagnostics.append(
                    {"source": source, "decisionKind": kind, "error": str(lane_exc)}
                )
        deduped: dict[str, dict[str, Any]] = {}
        for item in combined:
            key = str(item.get("fingerprint") or item.get("challengeId"))
            incumbent = deduped.get(key)
            if incumbent is None or float(item.get("publishability", 0.0)) > float(
                incumbent.get("publishability", 0.0)
            ):
                deduped[key] = item
        result = sorted(
            deduped.values(),
            key=lambda item: (
                float(item.get("publishability", 0.0)),
                float(item.get("difficulty", 0.0)),
            ),
            reverse=True,
        )[: request.maximumResults]
        STORE.update(
            job_id,
            status="completed",
            progress=1.0,
            result={
                "challenges": result,
                "count": len(result),
                "diagnostics": {"lanes": diagnostics},
            },
        )
    except Exception as exc:
        STORE.update(
            job_id,
            status="failed",
            progress=1.0,
            error=f"{exc}\n{traceback.format_exc(limit=6)}",
        )


def _execute_history_replay_render_job(
    job_id: str, identifier: str, request_data: dict[str, Any]
) -> None:
    """Render one completed archived game exactly as it happened.

    The audit recommendation is an overlay only. The underlying continuation
    remains the hash-verified historical replay, and every bid/card decision is
    linked to a timecode in the sidecar manifest.
    """
    try:
        request = HistoricalReplayRenderRequest.model_validate(request_data)
        STORE.update(job_id, status="running", progress=0.05)
        game = LEARNING_STORE.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        decision_id = request.decisionId
        if decision_id is None:
            report = AUDIT.audit_game(
                identifier,
                mode="progressive",
                force=request.rerunAudit,
                maximum_decisions=request.maximumAuditDecisions,
                overrides={
                    "requested_samples": request.auditSamples,
                    "exact_allocation_limit": 10_000,
                    "exact_search_remaining": 9,
                    "use_conventions": False,
                    "seed": 1,
                },
            )
            candidates = [
                item
                for item in report.get("decisions", [])
                if item.get("decisionKind") == "play" and not item.get("forced")
            ]
            if not candidates:
                candidates = [
                    item
                    for item in report.get("decisions", [])
                    if item.get("decisionKind") == "play"
                ]
            if not candidates:
                raise ValueError("archived game has no card-play decision to pause on")
            candidates.sort(
                key=lambda item: (
                    bool(item.get("catastrophicCrossing")),
                    float(item.get("winProbabilityDrop", 0.0)),
                    float(item.get("expectedRegret", 0.0)),
                    float(item.get("debateScore", 0.0)),
                ),
                reverse=True,
            )
            decision_id = str(candidates[0]["decisionId"])
        else:
            AUDIT.audit_decision(
                decision_id,
                deep=True,
                force=request.rerunAudit,
                overrides={"requested_samples": request.auditSamples},
            )
        STORE.update(job_id, progress=0.32)
        episode = LEARNING.create_historical_replay_episode(
            identifier, decision_id=decision_id
        )
        profile = copy.deepcopy(ACTIVE_BUNDLE.video)
        themes = profile.get("themes", {})
        if request.theme in themes:
            profile["colors"] = copy.deepcopy(themes[request.theme])
        profile["selected_theme"] = request.theme
        profile["render"]["width"] = request.width
        profile["render"]["height"] = request.height
        profile["render"]["fps"] = request.fps
        profile["render"]["crf"] = request.crf
        plugin_id = (
            "premium_video"
            if request.rendererPlugin == "auto"
            else request.rendererPlugin
        )
        plugin = REGISTRY.get("video_renderer", plugin_id)
        if "historical_actual" not in plugin.descriptor.capabilities:
            raise ValueError(
                f"renderer {plugin_id} does not support historical audit playback"
            )
        STORE.update(job_id, progress=0.55)
        basename = f"replay-{str(game.get('serial') or game.get('gameId')).lower()}-{decision_id.lower()}"
        artifacts = plugin.execute(
            episode,
            export_preset="hooked_flagship",
            output_dir=OUTPUTS,
            basename=basename,
            video_profile=profile,
        )
        records = _record_render_artifacts(
            artifacts,
            episode,
            owner_kind="game",
            owner_id=str(game.get("gameHash") or game.get("gameId")),
        )
        result = {
            "gameHash": game.get("gameHash"),
            "gameSerial": game.get("serial"),
            "decisionId": decision_id,
            "auditId": episode.audit_id,
            "playbackMode": episode.playback_mode,
            "video": f"/outputs/{artifacts.video.name}",
            "poster": f"/outputs/{artifacts.poster.name}",
            "episodeJson": f"/outputs/{artifacts.episode_json.name}",
            "captions": f"/outputs/{artifacts.captions.name}",
            "captionsVtt": f"/outputs/{artifacts.captions_vtt.name}"
            if artifacts.captions_vtt
            else None,
            "transcript": f"/outputs/{artifacts.transcript.name}"
            if artifacts.transcript
            else None,
            "contactSheet": f"/outputs/{artifacts.contact_sheet.name}",
            "manifest": f"/outputs/{artifacts.manifest.name}",
            "timeline": f"/outputs/{artifacts.timeline.name}"
            if artifacts.timeline
            else None,
            "videoRecords": [item.get("videoId") for item in records],
        }
        result["videoId"] = records[0].get("videoId") if records else None
        STORE.update(job_id, status="completed", progress=1.0, result=result)
    except Exception as exc:
        STORE.update(
            job_id,
            status="failed",
            progress=1.0,
            error=f"{exc}\n{traceback.format_exc(limit=6)}",
        )


def _execute_challenge_render_job(
    job_id: str, challenge_id: str, request_data: dict[str, Any]
) -> None:
    try:
        request = ChallengeRenderRequest.model_validate(request_data)
        STORE.update(job_id, status="running", progress=0.08)
        episode = LEARNING_STORE.load_challenge_episode(challenge_id)
        if episode is None:
            raise KeyError(challenge_id)
        profile = copy.deepcopy(ACTIVE_BUNDLE.video)
        themes = profile.get("themes", {})
        if request.theme in themes:
            profile["colors"] = copy.deepcopy(themes[request.theme])
        profile["selected_theme"] = request.theme
        profile["render"]["width"] = request.width
        profile["render"]["height"] = request.height
        profile["render"]["fps"] = request.fps
        profile["render"]["crf"] = request.crf
        plugin_id = request.rendererPlugin
        if plugin_id == "auto":
            plugin_id = "premium_video"
        plugin = REGISTRY.get("video_renderer", plugin_id)
        STORE.update(job_id, progress=0.22)
        artifacts = plugin.execute(
            episode,
            export_preset="decision_clip",
            output_dir=OUTPUTS,
            basename=f"challenge-{challenge_id.lower()}",
            video_profile=profile,
        )
        video_records = _record_render_artifacts(
            artifacts, episode, owner_kind="challenge", owner_id=challenge_id
        )
        result = {
            "challengeId": challenge_id,
            "video": f"/outputs/{artifacts.video.name}",
            "poster": f"/outputs/{artifacts.poster.name}",
            "episodeJson": f"/outputs/{artifacts.episode_json.name}",
            "captions": f"/outputs/{artifacts.captions.name}",
            "captionsVtt": f"/outputs/{artifacts.captions_vtt.name}"
            if artifacts.captions_vtt
            else None,
            "transcript": f"/outputs/{artifacts.transcript.name}"
            if artifacts.transcript
            else None,
            "contactSheet": f"/outputs/{artifacts.contact_sheet.name}",
            "manifest": f"/outputs/{artifacts.manifest.name}",
            "timeline": f"/outputs/{artifacts.timeline.name}"
            if artifacts.timeline
            else None,
        }
        artifact_pairs = [
            ("video", artifacts.video),
            ("poster", artifacts.poster),
            ("episode", artifacts.episode_json),
            ("captions", artifacts.captions),
            ("contact_sheet", artifacts.contact_sheet),
            ("manifest", artifacts.manifest),
        ]
        if artifacts.timeline is not None:
            artifact_pairs.append(("timeline", artifacts.timeline))
        if getattr(artifacts, "captions_vtt", None) is not None:
            artifact_pairs.append(("captions_vtt", artifacts.captions_vtt))
        if getattr(artifacts, "transcript", None) is not None:
            artifact_pairs.append(("transcript", artifacts.transcript))
        artifact_pairs.extend(
            (f"semantic_scene_{index}", path)
            for index, path in enumerate(
                getattr(artifacts, "semantic_files", ()),
                start=1,
            )
        )
        for kind, path in artifact_pairs:
            LEARNING_STORE.link_artifact(
                owner_kind="challenge",
                owner_id=challenge_id,
                kind=kind,
                uri=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata={"jobId": job_id},
            )
        result["videoRecords"] = [item.get("videoId") for item in video_records]
        result["videoId"] = video_records[0].get("videoId") if video_records else None
        STORE.update(job_id, status="completed", progress=1.0, result=result)
    except Exception as exc:
        STORE.update(
            job_id,
            status="failed",
            progress=1.0,
            error=f"{exc}\n{traceback.format_exc(limit=6)}",
        )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/play", status_code=307)


@app.get("/studio", include_in_schema=False)
def studio():
    return RedirectResponse("/foundry", status_code=307)


@app.get("/foundry", include_in_schema=False)
def foundry_dashboard():
    if not _FOUNDRY_STATIC.exists():
        raise HTTPException(503, "Foundry dashboard assets are not built")
    template = (_FOUNDRY_STATIC / "index.html").read_text(encoding="utf-8")
    profile = dict(ACTIVE_BUNDLE.component("product")["foundry_copy"])
    serialized = (
        json.dumps(
            profile,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    rendered = template.replace("{{foundry_copy_json}}", serialized).replace(
        "{{foundry_page_title}}",
        html.escape(str(profile["page_title"]), quote=True),
    )
    if "{{foundry_copy_json}}" in rendered or "{{foundry_page_title}}" in rendered:
        raise HTTPException(503, "Foundry copy profile is incomplete")
    return HTMLResponse(rendered, headers={"Cache-Control": "no-store"})


@app.get("/api/v1/foundry-copy")
def foundry_copy():
    profile = ACTIVE_BUNDLE.components["product"]
    copy_profile = dict(ACTIVE_BUNDLE.component("product")["foundry_copy"])
    return {
        "schemaVersion": 1,
        "locale": copy_profile["locale"],
        "direction": copy_profile["direction"],
        "profileVersion": profile.version,
        "profileSha256": profile.sha256,
        "copy": copy_profile,
    }


@app.get("/guide", include_in_schema=False)
def engine_guide_page():
    if not _GUIDE_STATIC.exists():
        raise HTTPException(503, "Engine guide assets are unavailable")
    template = (_GUIDE_STATIC / "index.html").read_text(encoding="utf-8")
    serialized = (
        json.dumps(
            ENGINE_GUIDE,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    rendered = (
        template.replace("{{engine_guide_json}}", serialized)
        .replace(
            "{{guide_page_title}}",
            html.escape(str(ENGINE_GUIDE["pageTitle"]), quote=True),
        )
        .replace("{{guide_asset_version}}", str(ENGINE_GUIDE["guideHash"]))
    )
    if (
        "{{engine_guide_json}}" in rendered
        or "{{guide_page_title}}" in rendered
        or "{{guide_asset_version}}" in rendered
    ):
        raise HTTPException(503, "Engine guide profile is incomplete")
    return HTMLResponse(rendered, headers={"Cache-Control": "no-store"})


@app.get("/api/v1/engine-guide", response_model=EngineGuideResponse)
def engine_guide():
    return ENGINE_GUIDE


@app.get("/learn", include_in_schema=False)
def learning_studio():
    return FileResponse(
        ROOT / "src/baloot_platform/service/static/learning_studio.html"
    )


@app.get("/replay", include_in_schema=False)
def replay_studio():
    return FileResponse(ROOT / "src/baloot_platform/service/static/replay_studio.html")


@app.get("/challenges", include_in_schema=False)
def challenge_studio():
    return RedirectResponse("/foundry", status_code=307)


@app.get("/decision-workbench", include_in_schema=False)
def decision_workbench():
    return RedirectResponse("/foundry", status_code=307)


@app.get("/hitl-studio", include_in_schema=False)
def hitl_studio():
    return RedirectResponse("/foundry", status_code=307)


@app.get("/play", include_in_schema=False)
def player_app():
    player_root = ROOT / "src/baloot_platform/service/static/player"
    template = (player_root / "index.html").read_text(encoding="utf-8")
    asset_digest = hashlib.sha256()
    for asset_name in ("app.css", "app.js"):
        asset_digest.update(asset_name.encode("utf-8"))
        asset_digest.update((player_root / asset_name).read_bytes())
    asset_version = asset_digest.hexdigest()[:12]
    governed_copy = dict(ACTIVE_BUNDLE.video["challenge_copy"])
    serialized = (
        json.dumps(
            governed_copy,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    rendered = template.replace("{{player_copy_json}}", serialized).replace(
        "{{player_asset_version}}", asset_version
    )
    for key, value in governed_copy.items():
        rendered = rendered.replace(
            "{{" + str(key) + "}}",
            html.escape(str(value), quote=True),
        )
    if "{{" in rendered or "}}" in rendered:
        raise HTTPException(503, "player copy profile is incomplete")
    return HTMLResponse(rendered, headers={"Cache-Control": "no-store"})


@app.get("/api/v1/challenge-copy")
def challenge_copy():
    profile = ACTIVE_BUNDLE.components["video"]
    return {
        "schemaVersion": 1,
        "locale": ACTIVE_BUNDLE.video["render"]["locale"],
        "profileVersion": profile.version,
        "profileSha256": profile.sha256,
        "copy": dict(ACTIVE_BUNDLE.video["challenge_copy"]),
    }


@app.get("/c/{code}", include_in_schema=False)
def public_duplicate_challenge(code: str):
    if not _CHALLENGE_STATIC.exists():
        raise HTTPException(503, "challenge player assets are unavailable")
    template = (_CHALLENGE_STATIC / "index.html").read_text(encoding="utf-8")
    governed_copy = dict(ACTIVE_BUNDLE.video["challenge_copy"])
    serialized = (
        json.dumps(
            governed_copy,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    rendered = template.replace("{{challenge_copy_json}}", serialized)
    rendered = rendered.replace(
        "{{page_title}}",
        html.escape(str(governed_copy["page_title"]), quote=True),
    )
    if "{{" in rendered or "}}" in rendered:
        raise HTTPException(503, "challenge copy profile is incomplete")
    return HTMLResponse(rendered, headers={"Cache-Control": "no-store"})


@app.get("/healthz")
@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/readyz")
def ready():
    bundle = ACTIVE_BUNDLE
    return {
        "status": "ready",
        "bundleHash": bundle.sha256,
        "ffmpeg": bool(__import__("shutil").which("ffmpeg")),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [
        "# TYPE baloot_http_requests_total counter",
        "# TYPE baloot_http_latency_microseconds_total counter",
    ]
    with _METRIC_LOCK:
        items = list(_METRICS.items())
    for (method, path, status), value in sorted(items):
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        if method == "LATENCY_US":
            lines.append(
                f'baloot_http_latency_microseconds_total{{path="{escaped}"}} {value}'
            )
        else:
            lines.append(
                f'baloot_http_requests_total{{method="{method}",path="{escaped}",status="{status}"}} {value}'
            )
    lines.append("")
    return "\n".join(lines)


@app.get("/api/v1/options")
def options():
    bundle = ACTIVE_BUNDLE
    return {
        "bundle": bundle.to_dict(),
        "plugins": REGISTRY.list(),
        "rulesStatus": {
            "expertRatified": bool(
                bundle.rules.data["profile"].get("expert_ratified", False)
            ),
            "status": bundle.rules.data["profile"].get("status"),
        },
        "profiles": {
            "rules": ["full", "sun_only", "hokum_only"],
            "conventions": ["hard_facts_only", "experimental_weighting"],
        },
        "cardPacks": CARD_PACKS.options(),
        "pacingProfiles": ["fast", "broadcast", "authentic", "cinematic"],
        "legacyVideoCompatibility": {
            "deprecated": True,
            "authority": "research_v4",
            "requiredFields": ["bundleRevisionId"],
            "requiredHeaders": ["Idempotency-Key"],
            "supportedScopes": ["round"],
            "supportedRulesProfiles": ["full"],
            "successorEndpoints": copy.deepcopy(_LEGACY_VIDEO_V4_SUCCESSORS),
        },
    }


@app.post(
    "/api/v1/video-jobs",
    status_code=202,
    deprecated=True,
    openapi_extra=_LEGACY_VIDEO_V4_CREATE_OPENAPI,
)
def create_video_job(body: VideoJobRequest, request: Request):
    _require_trusted(request)
    idempotency_key = request.headers.get("idempotency-key", "").strip()
    if not idempotency_key:
        raise HTTPException(
            428,
            "Idempotency-Key header is required",
            headers={"Deprecation": "true"},
        )
    try:
        payload = _submit_legacy_video_compatibility(
            body,
            idempotency_key=idempotency_key,
        )
    except LegacyVideoCompatibilityError as exc:
        raise HTTPException(
            422,
            exc.diagnostic,
            headers={"Deprecation": "true"},
        ) from exc
    except ResearchLaunchAdmissionError as exc:
        raise HTTPException(
            409,
            exc.diagnostic,
            headers={"Deprecation": "true"},
        ) from exc
    except ResearchMutationConflictError as exc:
        raise HTTPException(
            409,
            str(exc),
            headers={"Deprecation": "true"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            422,
            str(exc),
            headers={"Deprecation": "true"},
        ) from exc
    return JSONResponse(
        status_code=202,
        content=payload,
        headers={
            "Deprecation": "true",
            "Link": '</api/v2/research-recipes>; rel="successor-version"',
        },
    )


@app.get(
    "/api/v1/video-jobs/{job_id}",
    deprecated=True,
    openapi_extra=_LEGACY_VIDEO_V4_OPENAPI,
)
def get_video_job(job_id: str, request: Request):
    job = STORE.get(job_id)
    if job is not None:
        return JSONResponse(
            content=job,
            headers={
                "Deprecation": "true",
                "Link": '</api/v2/research-results>; rel="successor-version"',
            },
        )
    if not job_id.startswith("RUN-"):
        raise HTTPException(404, "job not found")
    _require_trusted(request)
    run = RESEARCH.get_run(job_id)
    if run is None or not _legacy_video_run_is_v4(run):
        raise HTTPException(404, "job not found")
    return JSONResponse(
        content=_legacy_video_run_view(run),
        headers={
            "Deprecation": "true",
            "Link": '</api/v2/research-runs>; rel="successor-version"',
        },
    )


@app.post("/api/v1/contexts:validate")
def validate_context(request: ContextRequest, http_request: Request):
    _require_trusted(http_request)
    bundle = _bundle_for_mode(request.rulesProfile)
    state = replay_trusted_context(request.state, bundle.rules)
    return {
        "valid": True,
        "state": state.to_dict(reveal_hands=True),
        "bundleHash": bundle.sha256,
    }


@app.post("/api/v1/contexts:continue")
def continue_context(request: ContextRequest, http_request: Request):
    _require_trusted(http_request)
    bundle = _bundle_for_mode(request.rulesProfile)
    state = continue_trusted_context(request.state, bundle.rules)
    return {"state": state.to_dict(reveal_hands=True), "bundleHash": bundle.sha256}


@app.post("/api/v1/games", status_code=201)
def create_game(request: GameCreateRequest):
    game = GAMES.create(
        seed=request.seed,
        dealer=Seat.parse(request.dealer),
        mode=request.rulesProfile,
        match_score=(request.matchScoreUs, request.matchScoreThem),
    )
    return game.to_dict(reveal_hands=False)


@app.get("/api/v1/games/{game_id}")
def get_game(game_id: str, request: Request, revealHands: bool = False):
    if revealHands:
        _require_trusted(request)
    try:
        return GAMES.get(game_id).to_dict(reveal_hands=revealHands)
    except KeyError:
        raise HTTPException(404, "game not found")


@app.post("/api/v1/games/{game_id}/bids")
def submit_bid(game_id: str, request: BidRequest):
    try:
        game = GAMES.bid(
            game_id,
            seat=Seat.parse(request.seat),
            action=request.action,
            trump=_parse_suit(request.trumpSuit) if request.trumpSuit else None,
            expected_version=request.expectedVersion,
        )
        return game.to_dict(reveal_hands=False)
    except KeyError:
        raise HTTPException(404, "game not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/v1/games/{game_id}/actions")
def play_game_card(game_id: str, request: PlayRequest):
    try:
        game = GAMES.play(
            game_id,
            seat=Seat.parse(request.seat),
            card=request.card,
            ekka=request.ekka,
            expected_version=request.expectedVersion,
        )
        return game.to_dict(reveal_hands=False)
    except KeyError:
        raise HTTPException(404, "game not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/v1/games/{game_id}/bots:advance")
def advance_game_bots(
    game_id: str, stopBefore: str | None = "south", maximumActions: int = 64
):
    try:
        stop = Seat.parse(stopBefore) if stopBefore else None
        return GAMES.advance_bots(
            game_id, stop_before=stop, maximum_actions=maximumActions
        ).to_dict(reveal_hands=False)
    except KeyError:
        raise HTTPException(404, "game not found")


@app.post("/api/v1/games/{game_id}/rounds:next")
def next_game_round(game_id: str, seed: int | None = None):
    try:
        return GAMES.next_round(game_id, seed=seed).to_dict(reveal_hands=False)
    except KeyError:
        raise HTTPException(404, "game not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/v1/bundles/current")
def current_bundle(mode: str = "full"):
    return _bundle_for_mode(mode).to_dict()


@app.get("/api/v1/games/{game_id}/audit")
def game_audit(game_id: str):
    try:
        GAMES.get(game_id)
    except KeyError:
        raise HTTPException(404, "game not found")
    return {"gameId": game_id, "snapshots": GAME_STORE.audit(game_id)}


@app.post("/api/v1/games/{game_id}/deal:annul")
def annul_game_deal(game_id: str, request: AnnulRequest):
    try:
        game = GAMES.get(game_id)
        GameManager._version(game, request.expectedVersion)
        game.engine.annul_initial_deal(
            game.state, Seat.parse(request.seat), reason=request.reason
        )
        game.version += 1
        GAMES._persist(game)
        return game.to_dict(reveal_hands=False)
    except KeyError:
        raise HTTPException(404, "game not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/v1/analyses")
def run_analysis(request: AnalysisRequest, http_request: Request):
    _require_trusted(http_request)
    from baloot_platform.analysis.evaluator import analyze
    from baloot_platform.analysis.models import InformationState

    bundle = _bundle_for_mode(request.rulesProfile)
    complete = replay_trusted_context(request.state, bundle.rules)
    observer = Seat.parse(request.observer)
    if complete.current_player != observer:
        raise HTTPException(422, "observer must be the current player")
    info = InformationState.from_complete(complete, observer)
    convention = copy.deepcopy(bundle.convention)
    convention["apply_to_probability"] = request.useExperimentalConventions
    try:
        if request.searchMethod == "information_set_mcts":
            from baloot_platform.analysis.ismcts_v8 import analyze_information_set_mcts
            from baloot_platform.core.engine import BalootEngine

            result = analyze_information_set_mcts(
                info,
                BalootEngine(bundle.rules),
                convention_profile=convention,
                particles=request.ismctsParticles,
                exact_allocation_limit=request.exactAllocationLimit,
                iterations=request.ismctsIterations,
                exploration=request.ismctsExploration,
                seed=request.seed,
            )
        elif request.searchMethod == "hybrid_team_public":
            from baloot_platform.analysis.team_public_search_v10 import (
                analyze_team_public_hybrid,
            )
            from baloot_platform.core.engine import BalootEngine

            result = analyze_team_public_hybrid(
                info,
                BalootEngine(bundle.rules),
                convention_profile=convention,
                particles=request.ismctsParticles,
                exact_allocation_limit=request.exactAllocationLimit,
                exact_remaining_threshold=request.exactSearchRemaining,
                ismcts_iterations=request.ismctsIterations,
                exploration=request.ismctsExploration,
                seed=request.seed,
            )
        else:
            result = analyze(
                info,
                bundle.rules,
                convention,
                requested_samples=request.samples,
                exact_allocation_limit=request.exactAllocationLimit,
                exact_search_max_remaining=request.exactSearchRemaining,
                maximum_sampling_attempts=hidden_deal_attempt_budget(
                    request.samples,
                    multiplier=1000,
                    minimum=10_000,
                ),
                seed=request.seed,
            )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "bundleHash": bundle.sha256,
        "informationState": info.to_dict(),
        "searchMethod": request.searchMethod,
        "analysis": result.to_dict(),
    }


@app.post(
    "/api/v1/video-jobs/{job_id}:cancel",
    deprecated=True,
    openapi_extra=_LEGACY_VIDEO_V4_OPENAPI,
)
def cancel_video_job(job_id: str, request: Request):
    if STORE.get(job_id) is not None:
        if not STORE.request_cancel(job_id):
            raise HTTPException(409, "job cannot be cancelled")
        return JSONResponse(
            content={"jobId": job_id, "status": "cancel_requested"},
            headers={"Deprecation": "true"},
        )
    if not job_id.startswith("RUN-"):
        raise HTTPException(404, "job not found")
    _require_trusted(request)
    try:
        existing = RESEARCH.get_run(job_id)
        if existing is None or not _legacy_video_run_is_v4(existing):
            raise KeyError(job_id)
        run = RESEARCH.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return JSONResponse(
        content=_legacy_video_run_view(run),
        headers={
            "Deprecation": "true",
            "Link": '</api/v2/research-runs>; rel="successor-version"',
        },
    )


@app.get(
    "/api/v1/video-jobs/{job_id}/events",
    deprecated=True,
    openapi_extra=_LEGACY_VIDEO_V4_OPENAPI,
)
def video_job_events(job_id: str, request: Request):
    if STORE.get(job_id) is not None:
        return JSONResponse(
            content={"jobId": job_id, "events": STORE.events(job_id)},
            headers={"Deprecation": "true"},
        )
    if not job_id.startswith("RUN-"):
        raise HTTPException(404, "job not found")
    _require_trusted(request)
    try:
        run = RESEARCH.get_run(job_id)
        if run is None or not _legacy_video_run_is_v4(run):
            raise KeyError(job_id)
        events = _operator_research_run_v1({"events": RESEARCH.events(job_id)})["events"]
    except KeyError:
        raise HTTPException(404, "job not found")
    return JSONResponse(
        content={
            "jobId": job_id,
            "authority": "research_v4",
            "events": events,
            "deprecation": {
                "schemaVersion": 1,
                "status": "deprecated",
                "successorEndpoints": copy.deepcopy(_LEGACY_VIDEO_V4_SUCCESSORS),
            },
        },
        headers={
            "Deprecation": "true",
            "Link": '</api/v2/research-runs>; rel="successor-version"',
        },
    )


class ConfigurationValidateRequest(BaseModel):
    request: dict[str, Any]


@app.post("/api/v1/configuration:validate")
def validate_configuration(request: ConfigurationValidateRequest):
    candidate = VideoJobRequest.model_validate(request.request)
    rules, convention, video, bundle_hash = _rules_for(candidate)
    try:
        plugin = REGISTRY.get("video_renderer", candidate.rendererPlugin)
    except KeyError as exc:
        raise HTTPException(422, str(exc)) from exc
    if (
        candidate.scope == "match"
        and "multi_round_match" not in plugin.descriptor.capabilities
    ):
        raise HTTPException(422, "selected renderer does not support match scope")
    if (
        candidate.scope == "round"
        and "multi_round_match" in plugin.descriptor.capabilities
    ):
        raise HTTPException(422, "selected renderer requires match scope")
    return {
        "valid": True,
        "bundleHash": bundle_hash,
        "rulesHash": rules.sha256,
        "resolved": {
            "scope": candidate.scope,
            "contracts": rules.data["contracts"]["allowed"],
            "firstLeader": rules.data["play"]["first_leader"],
            "projects": rules.data["projects"]["enabled"],
            "matchTarget": rules.data["scoring"]["match"]["target"],
            "conventionsApplied": bool(convention.get("apply_to_probability", False)),
            "renderer": candidate.rendererPlugin,
            "theme": candidate.theme,
            "resolution": [candidate.width, candidate.height],
        },
    }


@app.get("/api/v1/plugins-legacy", include_in_schema=False)
def plugins_legacy():
    return {"plugins": REGISTRY.list(), "deprecated": True}


@app.get("/api/v1/observability/snapshot")
def observability_snapshot():
    return TELEMETRY.snapshot()


@app.get("/api/v1/capabilities")
def capabilities():
    path = ROOT / "generated/capability-parity-v19.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": __version__,
        "status": "not_generated",
        "plugins": len(REGISTRY.list()),
    }


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/v1/agents/profiles")
def agent_profiles():
    return {
        "defaultProfile": AGENT_REGISTRY.default_profile,
        "defaultLineup": AGENT_REGISTRY.default_lineup,
        "profiles": AGENT_REGISTRY.list(),
    }


@app.get("/api/v1/games")
def list_live_games(search: str = "", limit: int = 100):
    return {"games": GAMES.list(query=search, limit=limit)}


@app.get("/api/v1/games/{game_id}/versions")
def game_versions(game_id: str):
    try:
        return {"gameId": game_id, "versions": GAMES.versions(game_id)}
    except KeyError:
        raise HTTPException(404, "game not found")


@app.get("/api/v1/games/{game_id}/decisions")
def game_decisions(game_id: str, request: Request, limit: int = 1000):
    decisions = LEARNING_STORE.list_decisions(game_id, limit=limit)
    return {
        "gameId": game_id,
        "decisions": [
            _project_decision_for_request(decision, request) for decision in decisions
        ],
    }


@app.post("/api/v1/games/{game_id}:archive")
def archive_live_game(game_id: str):
    try:
        game = GAMES.get(game_id)
    except KeyError:
        raise HTTPException(404, "game not found")
    return LEARNING.archive_session(game, status="manual_archive")


@app.post("/api/v1/games/{game_id}/versions/{version}:branch", status_code=201)
def branch_game_version(game_id: str, version: int):
    try:
        return GAMES.branch_from_version(game_id, version).to_dict(reveal_hands=False)
    except KeyError:
        raise HTTPException(404, "game version not found")


@app.get("/api/v1/history/games")
def history_games(search: str = "", limit: int = 100):
    return {"games": LEARNING_STORE.list_games(query=search, limit=limit)}


@app.get("/api/v1/history/games/{identifier}")
def history_game(identifier: str):
    return _require_private_history_game(identifier)


@app.get("/api/v1/history/games/{identifier}/replay")
def replay_history_package(identifier: str):
    game = _require_private_history_game(identifier)
    game_hash = game.get("gameHash")
    game_id = game.get("gameId")
    return {
        "game": game,
        "verification": LEARNING_STORE.verify_game(identifier),
        "audits": LEARNING_STORE.list_game_audits(game_hash or game_id, limit=50),
        "videos": LEARNING_STORE.list_video_records(game_hash=game_hash, limit=100)
        if game_hash
        else LEARNING_STORE.list_video_records("game", str(game_id), limit=100),
        "decisions": game.get("decisions")
        or LEARNING_STORE.list_decisions(game_hash or game_id),
        "bundle": {
            "hash": game.get("bundleHash"),
            "engineVersion": game.get("engineVersion"),
        },
    }


@app.get("/api/v1/history/games/{identifier}/video-records")
def history_game_video_records(identifier: str):
    game = _require_private_history_game(identifier)
    game_hash = game.get("gameHash")
    values = (
        LEARNING_STORE.list_video_records(game_hash=game_hash, limit=100)
        if game_hash
        else LEARNING_STORE.list_video_records(
            "game", str(game.get("gameId")), limit=100
        )
    )
    return {"videos": values}


@app.get("/api/v1/history/games/{identifier}/verification")
def verify_history_game(identifier: str):
    _require_private_history_game(identifier)
    result = LEARNING_STORE.verify_game(identifier)
    if result.get("error") == "not found":
        raise HTTPException(404, "archived game not found")
    return result


@app.post("/api/v1/history/games/{identifier}:reopen", status_code=201)
def reopen_history_game(identifier: str, branch: bool = False):
    _require_private_history_game(identifier)
    try:
        return LEARNING.reopen_game(identifier, branch=branch)
    except KeyError:
        raise HTTPException(404, "archived game not found")


@app.get("/api/v1/history/decisions/{decision_id}")
def history_decision(decision_id: str, request: Request):
    item = LEARNING_STORE.get_decision(decision_id)
    if item is None:
        raise HTTPException(404, "decision not found")
    return _project_decision_for_request(item, request)


@app.post("/api/v1/history/decisions/{decision_id}:reanalyze")
def reanalyze_history_decision(
    decision_id: str,
    body: ReanalysisRequest,
    request: Request,
):
    _require_private_decision_access(decision_id, request)
    try:
        return LEARNING.reanalyze_decision(
            decision_id,
            samples=body.samples,
            exact_allocation_limit=body.exactAllocationLimit,
            exact_search_remaining=body.exactSearchRemaining,
            use_conventions=body.useExperimentalConventions,
            seed=body.seed,
        )
    except KeyError:
        raise HTTPException(404, "decision not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/v1/history/decisions/{decision_id}:fork", status_code=201)
def fork_history_decision(decision_id: str, body: ForkDecisionRequest, request: Request):
    _require_private_decision_access(decision_id, request)
    try:
        return LEARNING.fork_decision(
            decision_id,
            action=body.action,
            continue_to_end=body.continueToEnd,
            maximum_actions=body.maximumActions,
        )
    except KeyError:
        raise HTTPException(404, "decision not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/v1/history/decisions/{decision_id}/feedback", status_code=201)
def feedback_history_decision(decision_id: str, body: FeedbackRequest, request: Request):
    _require_private_decision_access(decision_id, request)
    try:
        return LEARNING.add_feedback(
            decision_id,
            preferred_action=body.preferredAction,
            confidence=body.confidence,
            rationale=body.rationale,
            source=body.source,
            reviewer=body.reviewer,
        )
    except KeyError:
        raise HTTPException(404, "decision not found")


@app.post("/api/v1/history/decisions/{decision_id}/intent", status_code=201)
def record_decision_intent(decision_id: str, body: FeedbackRequest, request: Request):
    """Record an actor/expert statement without pretending intent was inferred."""
    _require_private_decision_access(decision_id, request)
    try:
        return LEARNING.add_feedback(
            decision_id,
            preferred_action=body.preferredAction,
            confidence=body.confidence,
            rationale=body.rationale,
            source=body.source,
            reviewer=body.reviewer,
        )
    except KeyError:
        raise HTTPException(404, "decision not found")


@app.get("/api/v1/history/games/{identifier}/videos")
def history_game_videos(identifier: str, limit: int = 100):
    game = _require_private_history_game(identifier)
    game_hash = str(game.get("gameHash") or identifier)
    return {
        "videos": LEARNING_STORE.list_video_records(game_hash=game_hash, limit=limit)
    }


@app.get("/api/v1/history/decisions/{decision_id}/videos")
def history_decision_videos(decision_id: str, request: Request, limit: int = 100):
    _require_private_decision_access(decision_id, request)
    return {
        "videos": LEARNING_STORE.list_video_records(
            "decision", decision_id, limit=limit
        )
    }


@app.post("/api/v1/history/games/{identifier}/video-jobs", status_code=202)
def render_history_game(identifier: str, request: HistoricalReplayRenderRequest):
    game = _require_private_history_game(identifier)
    if str(game.get("status")) not in {"completed", "manual_archive", "branch_source"}:
        raise HTTPException(
            422, "historical replay rendering requires a completed archived round"
        )
    job_id = uuid.uuid4().hex[:16]
    STORE.create(
        job_id,
        {"identifier": identifier, **request.model_dump()},
        kind="history_replay_video",
    )
    EXECUTOR.submit(
        _execute_history_replay_render_job, job_id, identifier, request.model_dump()
    )
    return {"jobId": job_id, "status": "queued", "gameHash": game.get("gameHash")}


@app.post("/api/v1/history/decisions/{decision_id}/video-jobs", status_code=202)
def render_history_decision(
    decision_id: str,
    body: ChallengeRenderRequest,
    request: Request,
):
    _require_private_decision_access(decision_id, request)
    try:
        decision = LEARNING_STORE.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        if decision.get("decisionKind") == "bid":
            episode = LEARNING.create_bid_episode_from_decision(decision_id)
        else:
            episode = LEARNING.create_episode_from_decision(decision_id)
    except KeyError:
        raise HTTPException(404, "decision not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    fingerprint = hashlib.sha256(decision_id.encode("utf-8")).hexdigest()
    challenge_id = f"CHL-{fingerprint[:20].upper()}"
    episode = replace(episode, challenge_id=challenge_id)
    decision_kind = "bid" if decision.get("decisionKind") == "bid" else "play"
    LEARNING_STORE.put_challenge(
        challenge_id=challenge_id,
        decision_id=decision_id,
        fingerprint=fingerprint,
        archetype="history_reanalysis",
        publishability=0.5,
        payload={
            "challengeId": challenge_id,
            "source": "history",
            "decisionId": decision_id,
            "decision": {"kind": decision_kind},
        },
        episode=episode,
        status="candidate",
    )
    job_id = uuid.uuid4().hex[:16]
    STORE.create(
        job_id,
        {"challengeId": challenge_id, **body.model_dump()},
        kind="challenge_video",
    )
    EXECUTOR.submit(
        _execute_challenge_render_job, job_id, challenge_id, body.model_dump()
    )
    return {"jobId": job_id, "challengeId": challenge_id, "status": "queued"}


@app.post("/api/v1/history/games/{identifier}:audit")
def audit_history_game(identifier: str, request: ReplayAuditRequest):
    _require_private_history_game(identifier)
    try:
        report = AUDIT.audit_game(
            identifier,
            mode=request.mode,
            force=request.rerun,
            maximum_decisions=request.maximumDecisions,
            overrides={
                "requested_samples": request.samples,
                "exact_allocation_limit": request.exactAllocationLimit,
                "exact_search_remaining": request.exactSearchRemaining,
                "use_conventions": request.useExperimentalConventions,
                "seed": request.seed,
            },
        )
        return report
    except KeyError:
        raise HTTPException(404, "archived game not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/v1/history/games/{identifier}/audits")
def list_history_game_audits(identifier: str, limit: int = 50):
    _require_private_history_game(identifier)
    return {"audits": LEARNING_STORE.list_game_audits(identifier, limit=limit)}


@app.get("/api/v1/audits/{audit_id}")
def get_replay_audit(audit_id: str, request: Request):
    value = LEARNING_STORE.get_game_audit(audit_id)
    if value is None:
        raise HTTPException(404, "audit not found")
    decision_id = str(value.get("decisionId") or "")
    if decision_id:
        _require_private_decision_access(decision_id, request)
    else:
        _require_completed_shared_game_access(str(value.get("gameId") or ""))
        _require_completed_shared_analysis_decisions(value)
    return value


@app.post("/api/v1/history/decisions/{decision_id}:explain")
def explain_history_decision(
    decision_id: str,
    body: DecisionAuditRequest,
    request: Request,
):
    _require_private_decision_access(decision_id, request)
    try:
        value = AUDIT.explain_decision(
            decision_id,
            analysis_id=body.analysisId,
            rerun=body.rerun,
            overrides={
                "requested_samples": body.samples,
                "exact_allocation_limit": body.exactAllocationLimit,
                "exact_search_remaining": body.exactSearchRemaining,
                "use_conventions": body.useExperimentalConventions,
                "seed": body.seed,
            },
        )
        return value
    except KeyError:
        raise HTTPException(404, "decision not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/v1/history/decisions/{decision_id}/audits")
def list_history_decision_audits(decision_id: str, request: Request):
    _require_private_decision_access(decision_id, request)
    return {"audits": LEARNING_STORE.list_decision_audits(decision_id)}


@app.get("/api/v1/history/decisions/{decision_id}/comparison")
def compare_history_decision(decision_id: str, request: Request):
    _require_private_decision_access(decision_id, request)
    try:
        return AUDIT.compare_decision_versions(decision_id)
    except KeyError:
        raise HTTPException(404, "decision not found")


def _accepted_actions_for_decision(
    decision_id: str,
) -> tuple[list[Any], float, dict[str, Any]]:
    decision = LEARNING_STORE.get_decision(decision_id)
    if decision is None:
        raise KeyError(decision_id)
    audits = decision.get("audits") or LEARNING_STORE.list_decision_audits(decision_id)
    audit = audits[0] if audits else AUDIT.audit_decision(decision_id, deep=True)
    accepted = [
        alternative.get("action")
        for alternative in audit.get("alternatives", [])
        if float(alternative.get("regret", 0.0)) <= 1.5
        and float(alternative.get("winProbabilityDrop", 0.0)) <= 0.02
    ]
    if not accepted:
        accepted = [audit.get("bestAction")]
    return accepted, float(audit.get("predictedDifficulty", 50.0)), audit


@app.post("/api/v1/history/decisions/{decision_id}/attempts", status_code=201)
def record_decision_attempt(
    decision_id: str,
    body: DifficultyAttemptRequest,
    request: Request,
):
    _require_private_decision_access(decision_id, request)
    try:
        accepted, engine_difficulty, audit = _accepted_actions_for_decision(decision_id)
    except KeyError:
        raise HTTPException(404, "decision not found")
    result = DIFFICULTY.record_attempt(
        decision_id=decision_id,
        player_id=body.playerId,
        chosen_action=body.chosenAction,
        accepted_actions=accepted,
        engine_difficulty=engine_difficulty,
        response_seconds=body.responseSeconds,
        confidence=body.confidence,
        reported_skill=body.reportedSkill,
        changed_answer=body.changedAnswer,
        source="replay_studio",
    )
    return {**result, "auditId": audit.get("auditId"), "acceptedActions": accepted}


@app.get("/api/v1/history/decisions/{decision_id}/difficulty")
def get_decision_difficulty(decision_id: str, request: Request):
    _require_private_decision_access(decision_id, request)
    try:
        _accepted, engine_difficulty, audit = _accepted_actions_for_decision(
            decision_id
        )
    except KeyError:
        raise HTTPException(404, "decision not found")
    item = LEARNING_STORE.get_difficulty_item(decision_id)
    if item is None:
        item = DIFFICULTY.recalibrate(decision_id, engine_difficulty=engine_difficulty)
    return {
        **item,
        "auditId": audit.get("auditId"),
        "engineDifficulty": engine_difficulty,
    }


def _accepted_actions_for_challenge(
    challenge: dict[str, Any],
) -> tuple[list[Any], float]:
    decision_id = challenge.get("decisionId")
    decision = LEARNING_STORE.get_decision(str(decision_id)) if decision_id else None
    difficulty = float((challenge.get("payload") or {}).get("difficulty", 50.0))
    if decision is None:
        payload = challenge.get("payload") or {}
        challenge_decision = payload.get("decision") or {}
        recommended = (
            challenge_decision.get("recommendedCard")
            if challenge_decision.get("kind") == "play"
            else challenge_decision.get("recommendedAction")
        )
        if recommended is None:
            raise ValueError("challenge has no governed accepted action")
        return [recommended], difficulty
    analyses = list(decision.get("analyses") or [])
    if not analyses:
        analysis = LEARNING.reanalyze_decision(str(decision_id), samples=48)
        result = analysis.get("result", {})
    else:
        result = analyses[0].get("result", {})
    values = result.get("actions") or result.get("candidates") or []
    if not values:
        return [decision.get("selectedAction")], float(
            result.get("difficulty", difficulty)
        )

    def action_of(item: dict[str, Any]) -> Any:
        return (
            item.get("card")
            if "card" in item
            else {"action": item.get("action"), "trumpSuit": item.get("trumpSuit")}
        )

    def round_margin(item: dict[str, Any]) -> float:
        return float(
            item.get("expectedRoundMargin", item.get("expectedMatchMargin", 0.0))
        )

    def supported_probability(item: dict[str, Any]) -> float:
        return float(
            item.get(
                "matchWinProbability",
                item.get(
                    "roundLeadProbability",
                    item.get("winProbability", item.get("teamWinProbability", 0.5)),
                ),
            )
        )

    ordered = sorted(
        values,
        key=lambda item: (supported_probability(item), round_margin(item)),
        reverse=True,
    )
    best_margin = round_margin(ordered[0])
    best_win = supported_probability(ordered[0])
    accepted = [
        action_of(item)
        for item in ordered
        if best_margin - round_margin(item) <= 1.5
        and best_win - supported_probability(item) <= 0.02
    ]
    return accepted or [action_of(ordered[0])], float(
        result.get("difficulty", difficulty)
    )


@app.post("/api/v1/challenges/{challenge_id}/attempts", status_code=201)
def record_challenge_attempt(challenge_id: str, request: DifficultyAttemptRequest):
    challenge = LEARNING_STORE.get_challenge(challenge_id)
    if challenge is None:
        raise HTTPException(404, "challenge not found")
    try:
        accepted, engine_difficulty = _accepted_actions_for_challenge(challenge)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return DIFFICULTY.record_attempt(
        decision_id=challenge_id,
        player_id=request.playerId,
        chosen_action=request.chosenAction,
        accepted_actions=accepted,
        engine_difficulty=engine_difficulty,
        response_seconds=request.responseSeconds,
        confidence=request.confidence,
        reported_skill=request.reportedSkill,
        changed_answer=request.changedAnswer,
    )


@app.get("/api/v1/challenges/{challenge_id}/difficulty")
def get_challenge_difficulty(challenge_id: str):
    challenge = LEARNING_STORE.get_challenge(challenge_id)
    if challenge is None:
        raise HTTPException(404, "challenge not found")
    item = LEARNING_STORE.get_difficulty_item(challenge_id)
    if item is None:
        _, engine_difficulty = _accepted_actions_for_challenge(challenge)
        item = DIFFICULTY.recalibrate(challenge_id, engine_difficulty=engine_difficulty)
    return item


@app.post("/api/v1/challenges/{challenge_id}:recalibrate")
def recalibrate_challenge_difficulty(challenge_id: str):
    challenge = LEARNING_STORE.get_challenge(challenge_id)
    if challenge is None:
        raise HTTPException(404, "challenge not found")
    _, engine_difficulty = _accepted_actions_for_challenge(challenge)
    return DIFFICULTY.recalibrate(challenge_id, engine_difficulty=engine_difficulty)


@app.get("/api/v1/adaptive/challenges/next")
def adaptive_next_challenge(playerId: str | None = None, recent: str = ""):
    recent_ids = {value.strip() for value in recent.split(",") if value.strip()}
    value = DIFFICULTY.adaptive_next(playerId, recent_ids=recent_ids)
    if value is None:
        raise HTTPException(404, "no published challenge is available")
    return value


@app.get("/api/v1/videos/{video_id}")
def get_video_record(video_id: str):
    value = LEARNING_STORE.get_video_record(video_id)
    if value is None:
        raise HTTPException(404, "video record not found")
    return value


@app.get("/api/v1/videos/{video_id}/timeline")
def get_video_timeline(video_id: str):
    value = LEARNING_STORE.get_video_record(video_id)
    if value is None:
        raise HTTPException(404, "video record not found")
    return value.get("timeline") or {"segments": []}


@app.get("/api/v1/videos/{video_id}/audit-map")
def video_audit_map(video_id: str):
    """Return the immutable video timeline enriched with decision audits."""
    try:
        return build_video_audit_map(LEARNING_STORE, video_id)
    except KeyError:
        raise HTTPException(404, "video record not found")


@app.get("/api/v1/learning/queue")
def learning_queue(limit: int = 100):
    return {"items": LEARNING_STORE.learning_queue(limit=limit)}


@app.get("/api/v1/learning/datasets")
def learning_datasets():
    return {"datasets": LEARNING_STORE.list_datasets()}


@app.post("/api/v1/learning/datasets", status_code=201)
def create_learning_dataset(request: DatasetBuildRequest):
    try:
        return LEARNING.build_dataset(
            purpose=request.purpose, minimum_confidence=request.minimumConfidence
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/v1/learning/models")
def learning_models():
    return {"models": LEARNING_STORE.list_models()}


@app.post("/api/v1/learning/models:train", status_code=201)
def train_learning_model(request: ModelTrainRequest):
    overrides = {
        key: value
        for key, value in {
            "epochs": request.epochs,
            "learning_rate": request.learningRate,
            "l2": request.l2,
            "batch_size": request.batchSize,
            "seed": request.seed,
        }.items()
        if value is not None
    }
    try:
        dataset_id = request.datasetId
        if dataset_id is None:
            datasets = LEARNING_STORE.list_datasets()
            if not datasets:
                raise ValueError("create a reviewed dataset before training")
            dataset_id = str(datasets[0]["id"])
        return LEARNING.train_model(
            dataset_id, overrides=overrides, backend=request.backend
        )
    except KeyError:
        raise HTTPException(404, "dataset not found")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/v1/learning/models/{model_id}:evaluate")
def evaluate_learning_model(model_id: str, request: ModelEvaluateRequest):
    try:
        return LEARNING.evaluate_model(model_id, request.datasetId)
    except KeyError as exc:
        raise HTTPException(404, f"artifact not found: {exc}")


@app.post("/api/v1/learning/models/{model_id}:promote")
def promote_learning_model(model_id: str, request: ModelPromotionRequest):
    try:
        evidence = request.model_dump(exclude={"rollbackModelId"})
        return LEARNING.promote_model(
            model_id,
            shadow_evidence=evidence,
            rollback_model_id=request.rollbackModelId,
        )
    except KeyError:
        raise HTTPException(404, "model not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/v1/challenge-jobs", status_code=202)
def create_challenge_job(request: ChallengeMiningRequest):
    job_id = uuid.uuid4().hex[:16]
    STORE.create(job_id, request.model_dump(), kind="challenge_mining")
    EXECUTOR.submit(_execute_challenge_job, job_id, request.model_dump())
    return {"jobId": job_id, "status": "queued"}


def _require_local_foundry_execution() -> None:
    if os.environ.get("BALOOT_ENV", "development").strip().lower() not in {
        "development",
        "test",
    }:
        raise HTTPException(
            403,
            "local Foundry execution is available only in development or test mode",
        )


def _foundry_simulation_worker(lane: str, campaign_hash: str):
    common = {
        "worker_id": f"local-foundry-{campaign_hash[:12]}-{lane}",
        "lease_seconds": 900,
        "heartbeat_seconds": 60,
    }
    artifact_store = CONTAINER.learning.artifact_store
    if lane == "organic":
        return OrganicFoundryWorkerV1(
            FOUNDRY_CAMPAIGNS,
            artifact_store,
            REGISTRY.get("generator", "natural_laffa_generator_v1"),
            screening_profile=CONTAINER.organic_screening_profile,
            **common,
        )
    if lane == "guided":
        return GuidedFoundryWorkerV1(
            FOUNDRY_CAMPAIGNS,
            artifact_store,
            REGISTRY.get("generator", "guided_legal_search_v1"),
            **common,
        )
    if lane == "disagreement":
        return DisagreementFoundryWorkerV1(
            FOUNDRY_CAMPAIGNS,
            artifact_store,
            REGISTRY.get("generator", "observer_safe_disagreement_search_v1"),
            **common,
        )
    raise ValueError(f"unsupported generated Foundry lane: {lane}")


def _execute_local_foundry_campaign(campaign_hash: str) -> None:
    try:
        campaign = FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)
        if campaign is None:
            return
        workers = tuple(
            _foundry_simulation_worker(str(lane), campaign_hash)
            for lane in campaign["spec"]["lanes"]
        )
        while True:
            current = FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)
            if current is None or current["status"] in {
                "completed",
                "failed",
                "cancelled",
            }:
                return
            progressed = False
            for worker in workers:
                result = worker.run_once(campaign_hash=campaign_hash)
                progressed = progressed or result.status != "idle"
            if not progressed:
                shards = FOUNDRY_CAMPAIGNS.list_shards(campaign_hash)
                nonterminal = [
                    item for item in shards if item["status"] in {"queued", "running"}
                ]
                if not nonterminal:
                    return
                leases = [
                    float(item["leaseUntil"])
                    for item in nonterminal
                    if item["status"] == "running" and item.get("leaseUntil")
                ]
                delay = (
                    0.05
                    if any(item["status"] == "queued" for item in nonterminal)
                    else 1.0
                )
                if leases:
                    delay = min(delay, max(0.05, min(leases) - time.time()))
                time.sleep(delay)
    finally:
        with _LOCAL_FOUNDRY_LOCK:
            _LOCAL_FOUNDRY_CAMPAIGNS.discard(campaign_hash)


def _schedule_local_foundry_campaign(campaign_hash: str) -> None:
    with _LOCAL_FOUNDRY_LOCK:
        if campaign_hash in _LOCAL_FOUNDRY_CAMPAIGNS:
            return
        _LOCAL_FOUNDRY_CAMPAIGNS.add(campaign_hash)
    try:
        EXECUTOR.submit(_execute_local_foundry_campaign, campaign_hash)
    except BaseException:
        with _LOCAL_FOUNDRY_LOCK:
            _LOCAL_FOUNDRY_CAMPAIGNS.discard(campaign_hash)
        raise


def _execute_local_foundry_batch_fleet(campaign_hash: str) -> None:
    try:
        campaign = FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)
        if campaign is None or campaign["status"] in {
            "completed",
            "failed",
            "cancelled",
        }:
            return
        profile = dict(FOUNDRY_PROFILE["videoProductionBatchProfile"])
        shard_count = len(FOUNDRY_CAMPAIGNS.list_shards(campaign_hash))
        worker_count = max(1, min(int(profile["simulationWorkerCount"]), shard_count))
        timeout_seconds = int(profile["simulationTimeoutSeconds"])
        environment = dict(os.environ)
        source_root = str((ROOT / "src").resolve())
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = source_root + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        environment.update(
            {
                "BALOOT_FOUNDRY_CAMPAIGN_HASH": campaign_hash,
                "BALOOT_FOUNDRY_WORKER_COUNT": str(worker_count),
                "BALOOT_FOUNDRY_RUN_ID": f"batch-{campaign_hash[:16]}",
                "BALOOT_FOUNDRY_FLEET_TIMEOUT_SECONDS": str(timeout_seconds),
            }
        )
        completed = subprocess.run(
            [sys.executable, "-m", "baloot_platform.service.foundry_simulation_fleet"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0,
            check=False,
        )
        if completed.returncode != 0:
            # The fleet records fenced failure evidence before returning. A
            # bounded single-process retry preserves progress in local mode.
            _execute_local_foundry_campaign(campaign_hash)
    finally:
        with _LOCAL_FOUNDRY_LOCK:
            _LOCAL_FOUNDRY_CAMPAIGNS.discard(campaign_hash)


def _schedule_local_foundry_batch_campaign(campaign_hash: str) -> None:
    with _LOCAL_FOUNDRY_LOCK:
        if campaign_hash in _LOCAL_FOUNDRY_CAMPAIGNS:
            return
        _LOCAL_FOUNDRY_CAMPAIGNS.add(campaign_hash)
    try:
        EXECUTOR.submit(_execute_local_foundry_batch_fleet, campaign_hash)
    except BaseException:
        with _LOCAL_FOUNDRY_LOCK:
            _LOCAL_FOUNDRY_CAMPAIGNS.discard(campaign_hash)
        raise


def _execute_local_analysis_queue() -> None:
    global _LOCAL_ANALYSIS_WORKER_ACTIVE
    worker = FoundryDeepAnalysisWorkerV1(
        FOUNDRY_DEEP_ANALYSIS,
        CONTAINER.learning.artifact_store,
        worker_id=f"local-foundry-analysis-{os.getpid()}",
        lease_seconds=900,
        heartbeat_seconds=60,
    )
    try:
        while True:
            result = worker.run_once()
            if result.status != "idle":
                continue
            nonterminal = [
                item
                for status in ("queued", "running")
                for item in FOUNDRY_DEEP_ANALYSIS.list(status=status, limit=500)
            ]
            if not nonterminal:
                return
            leases = [
                float(item["leaseUntil"])
                for item in nonterminal
                if item["status"] == "running" and item.get("leaseUntil")
            ]
            delay = (
                0.05 if any(item["status"] == "queued" for item in nonterminal) else 1.0
            )
            if leases:
                delay = min(delay, max(0.05, min(leases) - time.time()))
            time.sleep(delay)
    finally:
        with _LOCAL_FOUNDRY_LOCK:
            _LOCAL_ANALYSIS_WORKER_ACTIVE = False


def _schedule_local_analysis_worker() -> None:
    global _LOCAL_ANALYSIS_WORKER_ACTIVE
    with _LOCAL_FOUNDRY_LOCK:
        if _LOCAL_ANALYSIS_WORKER_ACTIVE:
            return
        _LOCAL_ANALYSIS_WORKER_ACTIVE = True
    try:
        EXECUTOR.submit(_execute_local_analysis_queue)
    except BaseException:
        with _LOCAL_FOUNDRY_LOCK:
            _LOCAL_ANALYSIS_WORKER_ACTIVE = False
        raise


_VIDEO_PRODUCTION_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="baloot-video-production",
)
FOUNDRY_VIDEO_PRODUCTION = FoundryVideoBatchServiceV1(
    jobs=STORE,
    background_executor=_VIDEO_PRODUCTION_EXECUTOR,
    campaign_store=FOUNDRY_CAMPAIGNS,
    deep_analysis=FOUNDRY_DEEP_ANALYSIS,
    candidates=FOUNDRY_ANALYSIS_CANDIDATES,
    finalist_validations=FOUNDRY_FINALIST_VALIDATIONS,
    render_batches=RENDER_BATCHES,
    runtime_profile=FOUNDRY_PROFILE,
    schedule_campaign=_schedule_local_foundry_batch_campaign,
    schedule_analysis=_schedule_local_analysis_worker,
    qualified_library=FOUNDRY_CATALOGS,
    maximum_attempts=int(_FOUNDRY_JOB_POLICY["maximum_attempts"]),
)


def _recover_local_foundry_work() -> None:
    """Resume persisted work only in the single-node SQLite development mode."""

    for service in (
        RENDER_BATCHES,
        FOUNDRY_FINALIST_VALIDATIONS,
        CONTEXT_SEARCHES,
        SAKKA_GENERATIONS,
        FOUNDRY_VIDEO_PRODUCTION,
    ):
        recover = getattr(service, "recover_pending", None)
        if callable(recover):
            recover()

    cursor: str | None = None
    while True:
        campaigns = FOUNDRY_CAMPAIGNS.list_campaigns(after=cursor, limit=500)
        for campaign in campaigns:
            if campaign["status"] in {"queued", "running"}:
                _schedule_local_foundry_campaign(campaign["campaignHash"])
        if len(campaigns) < 500:
            break
        cursor = campaigns[-1]["campaignHash"]

    if any(
        FOUNDRY_DEEP_ANALYSIS.list(status=status, limit=1)
        for status in ("queued", "running")
    ):
        _schedule_local_analysis_worker()


@app.post(
    "/api/v2/video-production-batches",
    status_code=202,
    response_model=FoundryVideoProductionBatchJobResponse,
)
def create_foundry_video_production_batch(
    body: FoundryVideoProductionBatchCreateRequest,
    request: Request,
):
    _require_trusted(request)
    _require_local_foundry_execution()
    idempotency_key = request.headers.get("idempotency-key", "").strip()
    if not idempotency_key:
        raise HTTPException(428, "Idempotency-Key header is required")
    profile = dict(FOUNDRY_PROFILE["videoProductionBatchProfile"])
    if body.profileHash != profile.get("sha256"):
        raise HTTPException(409, "video production profile changed; refresh Foundry")
    if body.targetCount > int(profile["maximumTarget"]):
        raise HTTPException(422, "video production target exceeds the governed profile")
    try:
        return FOUNDRY_VIDEO_PRODUCTION.submit(
            {
                "schemaVersion": 1,
                "targetVideos": body.targetCount,
                "laneOrder": profile["laneOrder"],
                "seedStart": body.seedStart,
                "roundsPerWave": profile["waveSeedCount"],
                "maximumRounds": int(profile["waveSeedCount"])
                * int(profile["maximumWaves"]),
                "seedsPerShard": profile["seedsPerShard"],
                "finalistConcurrency": profile["finalistConcurrency"],
                "useCandidateResimulation": profile["useCandidateResimulation"],
                "renderConcurrency": profile["renderConcurrency"],
                "renderPolicy": profile["renderPolicy"],
                "exportPreset": "hooked_flagship",
                "revealPolicy": "after_commit",
                "allowedTruthLabels": profile["allowedTruthLabels"],
                "minimumSelectionScore": profile["minimumSelectionScore"],
                "minimumConfidence": profile["minimumConfidence"],
                "requirePlayDecision": profile["requirePlayDecision"],
                "requireCritical": profile["requireCritical"],
                "runtimeSnapshot": FoundryVideoBatchServiceV1.runtime_snapshot(
                    FOUNDRY_PROFILE
                ),
            },
            idempotency_key=idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get(
    "/api/v2/video-production-batches",
    response_model=FoundryVideoProductionBatchJobListResponse,
)
def list_foundry_video_production_batches(
    request: Request,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    return {"items": FOUNDRY_VIDEO_PRODUCTION.list(status=status, limit=limit)}


@app.get(
    "/api/v2/qualified-laffat",
    response_model=QualifiedLaffaLibraryResponse,
)
def list_qualified_laffat(
    request: Request,
    search: str = Query("", max_length=100),
    truthLabel: Literal["exact_optimal", "robust_best_known", "debatable"]
    | None = None,
    classification: str | None = Query(None, max_length=80),
    profile: str | None = Query(None, max_length=100),
    sort: Literal["laffa_score", "decision_score", "confidence"] = "laffa_score",
    direction: Literal["asc", "desc"] = "desc",
    cursor: str | None = Query(None, max_length=500),
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    try:
        return FOUNDRY_VIDEO_PRODUCTION.list_qualified_laffat(
            search=search,
            truth_label=truthLabel,
            classification=classification,
            profile=profile,
            sort=sort,
            direction=direction,
            cursor=cursor,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get(
    "/api/v2/video-production-batches/{job_id}",
    response_model=FoundryVideoProductionBatchJobResponse,
)
def get_foundry_video_production_batch(job_id: str, request: Request):
    _require_trusted(request)
    item = FOUNDRY_VIDEO_PRODUCTION.get(job_id)
    if item is None:
        raise HTTPException(404, "video production batch not found")
    return item


@app.post(
    "/api/v2/video-production-batches/{job_id}:resume",
    response_model=FoundryVideoProductionBatchJobResponse,
)
def resume_foundry_video_production_batch(job_id: str, request: Request):
    _require_trusted(request)
    _require_local_foundry_execution()
    try:
        return FOUNDRY_VIDEO_PRODUCTION.resume(job_id)
    except KeyError:
        raise HTTPException(404, "video production batch not found")


@app.post(
    "/api/v2/video-production-batches/{job_id}:cancel",
    response_model=FoundryVideoProductionBatchJobResponse,
)
def cancel_foundry_video_production_batch(job_id: str, request: Request):
    _require_trusted(request)
    item = FOUNDRY_VIDEO_PRODUCTION.get(job_id)
    if item is None:
        raise HTTPException(404, "video production batch not found")
    if not FOUNDRY_VIDEO_PRODUCTION.cancel(job_id):
        raise HTTPException(
            409, f"video production batch cannot be cancelled from {item['status']}"
        )
    return FOUNDRY_VIDEO_PRODUCTION.get(job_id)


@app.get("/api/v2/video-production-batches/{job_id}/events")
def get_foundry_video_production_batch_events(job_id: str, request: Request):
    _require_trusted(request)
    values = FOUNDRY_VIDEO_PRODUCTION.events(job_id)
    if values is None:
        raise HTTPException(404, "video production batch not found")
    return {"items": values}


@app.get(
    "/api/v2/domain-capabilities",
    response_model=ResearchDomainCapabilitiesResponse,
    response_model_exclude_unset=True,
)
def get_research_domain_capabilities(
    request: Request,
    includePayloads: bool = False,
    bundleRevisionId: str | None = None,
):
    _require_trusted(request)
    try:
        return RESEARCH.capabilities(
            include_payloads=includePayloads,
            bundle_revision_id=bundleRevisionId,
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get(
    "/api/v2/rule-primitives",
    response_model=ResearchRulePrimitiveListResponse,
    response_model_exclude_unset=True,
)
def list_research_rule_primitives(
    request: Request,
    bundleRevisionId: str | None = None,
):
    _require_trusted(request)
    try:
        capabilities = RESEARCH.capabilities(
            include_payloads=False,
            bundle_revision_id=bundleRevisionId,
        )
        response = {
            "schemaVersion": 1,
            "domainHash": capabilities["sha256"],
            "items": capabilities["rulePrimitives"],
        }
        if bundleRevisionId is not None:
            selection = capabilities.get("research", {}).get("configurationSelection")
            if (
                not isinstance(selection, dict)
                or selection.get("bundleRevisionId") != bundleRevisionId
                or selection.get("legacyDomainV2Hash") != capabilities["sha256"]
            ):
                raise ValueError(
                    "bundle-specific rule primitives lack an exact configuration selection"
                )
            response["configurationSelection"] = selection
        return response
    except (KeyError, PermissionError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/v2/domain-compilations/{compilation_hash}")
def get_domain_compilation(
    compilation_hash: str, request: Request, includePayloads: bool = False
):
    _require_trusted(request)
    item = RESEARCH.domain_compilation(
        compilation_hash, include_payloads=includePayloads
    )
    if item is None:
        raise HTTPException(404, "domain compilation not found")
    return item


@app.get("/api/v2/domain-compilations/{compilation_hash}/coverage")
def get_domain_compilation_coverage(compilation_hash: str, request: Request):
    _require_trusted(request)
    item = RESEARCH.domain_compilation_coverage(compilation_hash)
    if item is None:
        raise HTTPException(404, "domain compilation not found")
    return item


@app.post(
    "/api/v2/research-exact-traces:preview-legal-actions",
    response_model=ExactTraceLegalActionsPreviewV1Response,
)
def preview_research_exact_trace_legal_actions(
    body: ExactTraceLegalActionsPreviewV1Request,
    request: Request,
):
    _require_trusted(request)
    try:
        return RESEARCH.preview_exact_trace_legal_actions(
            body.model_dump(exclude_none=True)
        )
    except (KeyError, PermissionError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post(
    "/api/v2/research-recipes:validate",
    response_model=ResearchRecipeValidationResponse,
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
def validate_research_recipe(body: ResearchRecipeWireRequest, request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.validate_recipe(body.model_dump(exclude_none=True), new_authoring=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post(
    "/api/v2/research-recipes:estimate",
    response_model=ResearchRecipeEstimateResponse,
    response_model_exclude_unset=True,
)
def estimate_research_recipe(body: ResearchRecipeWireRequest, request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.estimate(body.model_dump(exclude_none=True), new_authoring=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post(
    "/api/v2/research-recipes",
    status_code=201,
    response_model=StoredResearchRecipeV4Response,
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
    responses={
        409: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key was reused for another intent.",
        },
        422: {
            "model": ResearchMutationValidationErrorResponse,
            "description": "Recipe or Idempotency-Key is invalid.",
        },
        428: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key is required.",
        },
    },
    openapi_extra=_RESEARCH_MUTATION_OPENAPI,
)
def create_research_recipe(
    body: ResearchRecipeCreateWireRequest,
    request: Request,
):
    _require_trusted(request)
    key = _require_research_mutation_key(request)
    try:
        return RESEARCH.create_recipe(
            body.model_dump(exclude_none=True),
            idempotency_key=key,
        )
    except ResearchMutationConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post(
    "/api/v2/research-recipes/{code}:preview-v4-conversion",
    response_model=ResearchRecipeV4ConversionPreviewResponse,
)
def preview_research_recipe_v4_conversion(
    code: str,
    body: ResearchRecipeV4ConversionPreviewRequest,
    request: Request,
):
    _require_trusted(request)
    payload = body.model_dump(exclude_none=True)
    try:
        preview = RESEARCH.preview_recipe_v4_conversion(
            code,
            decision_point=payload.get("decisionPoint"),
            outcome=payload.get("outcome"),
            proof=payload.get("proof"),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if preview is None:
        raise HTTPException(404, "research recipe not found")
    return preview


@app.get(
    "/api/v2/research-recipes",
    response_model=ResearchRecipeListResponse,
    response_model_exclude_unset=True,
)
def list_research_recipes(
    request: Request,
    status: Literal["draft", "published", "retired"] | None = None,
    cursor: str | None = Query(
        None,
        min_length=1,
        max_length=RECIPE_LIST_CURSOR_MAX_LENGTH,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    try:
        return RESEARCH.list_recipe_page(
            status=status,
            cursor=cursor,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get(
    "/api/v2/research-recipes/{code}",
    response_model=StoredResearchRecipeWireResponse,
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
def get_research_recipe(code: str, request: Request):
    _require_trusted(request)
    item = RESEARCH.get_recipe(code)
    if item is None:
        raise HTTPException(404, "research recipe not found")
    return item


@app.post(
    "/api/v2/research-recipes/{code}:retire",
    response_model=StoredResearchRecipeWireResponse,
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
    responses={
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research recipe was not found.",
        },
        409: {
            "model": ResearchMutationErrorResponse,
            "description": "Recipe state or Idempotency-Key conflicts.",
        },
        422: {
            "model": ResearchMutationValidationErrorResponse,
            "description": "Idempotency-Key is invalid.",
        },
        428: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key is required.",
        },
    },
    openapi_extra=_RESEARCH_MUTATION_OPENAPI,
)
def retire_research_recipe(
    code: str,
    request: Request,
):
    _require_trusted(request)
    key = _require_research_mutation_key(request)
    try:
        return RESEARCH.retire_recipe(code, idempotency_key=key)
    except KeyError:
        raise HTTPException(404, "research recipe not found")
    except ResearchMutationConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post(
    "/api/v2/research-runs",
    status_code=202,
    response_model=ResearchRunResponse,
    response_model_exclude_none=True,
    responses={
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research recipe was not found.",
        },
        409: {
            "model": ResearchRunConflictResponse,
            "description": "Research recipe state, run intent, or Estimate V2 launch admission conflicts.",
        },
        422: {
            "model": ResearchRequestValidationErrorResponse,
            "description": "Research run request is invalid.",
        },
        428: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key is required.",
        },
    },
    openapi_extra=_DURABLE_JOB_MUTATION_OPENAPI,
)
def create_research_run(body: ResearchRunCreateRequest, request: Request):
    _require_trusted(request)
    idempotency_key = _require_trimmed_idempotency_key(request)
    try:
        return RESEARCH.submit(body.recipeCode, idempotency_key=idempotency_key)
    except KeyError:
        raise HTTPException(404, "research recipe not found")
    except ResearchLaunchAdmissionError as exc:
        raise HTTPException(409, exc.diagnostic) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get(
    "/api/v2/research-runs",
    response_model=ResearchRunListResponse,
    response_model_exclude_none=True,
)
def list_research_runs(
    request: Request,
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    return {"items": RESEARCH.list_operator_runs(status=status, limit=limit)}


@app.get("/api/v2/research-stats", response_model=ResearchStatsResponse,
    responses={409: {"model": ResearchMutationErrorResponse,
        "description": "Retained performance evidence failed validation."}})
def get_research_statistics(request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.statistics()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get(
    "/api/v2/research-runs/{run_code}",
    response_model=ResearchRunResponse,
    response_model_exclude_none=True,
)
def get_research_run(run_code: str, request: Request):
    _require_trusted(request)
    item = RESEARCH.get_operator_run(run_code)
    if item is None:
        raise HTTPException(404, "research run not found")
    return item


@app.get(
    "/api/v2/research-runs/{run_code}/sequential-search-receipts",
    response_model=ResearchSequentialSearchReceiptListResponse,
)
def list_research_run_sequential_search_receipts(
    run_code: str,
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    if RESEARCH.get_run(run_code) is None:
        raise HTTPException(404, "research run not found")
    return RESEARCH.list_sequential_search_receipts(
        run_code,
        offset=offset,
        limit=limit,
    )


@app.get("/api/v2/research-runs/{run_code}/stats", response_model=ResearchStatsResponse,
    responses={409: {"model": ResearchMutationErrorResponse,
        "description": "Retained performance evidence failed validation."}})
def get_research_run_statistics(run_code: str, request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.statistics(run_code=run_code)
    except KeyError:
        raise HTTPException(404, "research run not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post(
    "/api/v2/research-runs/{run_code}:cancel",
    response_model=ResearchRunResponse,
    response_model_exclude_none=True,
    responses={
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research run was not found.",
        },
        409: {
            "model": ResearchMutationErrorResponse,
            "description": "Research run cannot be cancelled from its current state.",
        },
    },
)
def cancel_research_run(run_code: str, request: Request):
    _require_trusted(request)
    try:
        return _operator_research_run_v1(RESEARCH.cancel(run_code))
    except KeyError:
        raise HTTPException(404, "research run not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/research-runs/{run_code}:render-selected",
    status_code=202,
    response_model=FoundryRenderBatchJobResponse,
    responses={
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research run was not found.",
        },
        409: {
            "model": ResearchMutationErrorResponse,
            "description": "The frozen V4 run cannot prepare its requested videos.",
        },
        428: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key is required.",
        },
    },
    openapi_extra=_RENDER_BATCH_MUTATION_OPENAPI,
)
def render_selected_research_run(
    run_code: str,
    body: ResearchRenderPrepareRequest,
    request: Request,
):
    _require_trusted(request)
    idempotency_key = _require_trimmed_idempotency_key(
        request,
        maximum_length=200,
    )
    try:
        prepared = RESEARCH_RENDERS.prepare_selected_run(
            run_code,
            observer_seat=body.observerSeat,
            reveal_policy=body.revealPolicy,
            export_preset=body.exportPreset,
        )
        return RENDER_BATCHES.submit(
            tuple(prepared["items"]),
            idempotency_key=idempotency_key,
            playback_evidence=body.exportPreset == "hooked_flagship",
        )
    except KeyError:
        raise HTTPException(404, "research run not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get(
    "/api/v2/research-runs/{run_code}/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Research run server-sent event stream.",
            "content": {
                "text/event-stream": {"schema": {"type": "string"}},
            },
        },
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research run was not found.",
        },
    },
)
def stream_research_run_events(run_code: str, request: Request):
    _require_trusted(request)
    if RESEARCH.get_run(run_code) is None:
        raise HTTPException(404, "research run not found")
    try:
        initial_cursor = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        initial_cursor = 0

    def stream():
        cursor = initial_cursor
        previous_legacy: set[str] = set()
        initialize_legacy = cursor > 0
        while True:
            events = _operator_research_run_v1({"events": RESEARCH.events(run_code)})["events"]
            current_legacy: set[str] = set()
            for event in events:
                sequence = event.get("sequence")
                if type(sequence) is int:
                    if sequence <= cursor:
                        continue
                    event_id = sequence
                else:
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            event,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    current_legacy.add(fingerprint)
                    if initialize_legacy or fingerprint in previous_legacy:
                        continue
                    event_id = cursor + 1
                cursor = max(cursor, event_id)
                yield (
                    f"id: {event_id}\n"
                    f"event: {event['type']}\n"
                    f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
                )
            previous_legacy = current_legacy
            initialize_legacy = False
            item = RESEARCH.get_run(run_code)
            if item is None or item["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/v2/comparison-campaigns",
    status_code=202,
    response_model=ComparisonCampaignResponse,
    responses={
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research recipe was not found.",
        },
        409: {
            "model": ResearchMutationErrorResponse,
            "description": "Research recipe state or campaign intent conflicts.",
        },
        422: {
            "model": ResearchRequestValidationErrorResponse,
            "description": "Comparison campaign request is invalid.",
        },
        428: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key is required.",
        },
    },
    openapi_extra=_DURABLE_JOB_MUTATION_OPENAPI,
)
def create_comparison_campaign(body: ComparisonCampaignCreateRequest, request: Request):
    _require_trusted(request)
    idempotency_key = _require_trimmed_idempotency_key(request)
    try:
        return RESEARCH.create_comparison_campaign(
            body.recipeCode,
            idempotency_key=idempotency_key,
        )
    except KeyError:
        raise HTTPException(404, "research recipe not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/v2/comparison-campaigns", response_model=ComparisonCampaignListResponse)
def list_comparison_campaigns(request: Request, limit: int = Query(100, ge=1, le=500)):
    _require_trusted(request)
    return {"items": RESEARCH.list_comparison_campaigns(limit=limit)}


@app.get(
    "/api/v2/comparison-campaigns/{campaign_code}",
    response_model=ComparisonCampaignResponse,
)
def get_comparison_campaign(campaign_code: str, request: Request):
    _require_trusted(request)
    item = RESEARCH.get_comparison_campaign(campaign_code)
    if item is None:
        raise HTTPException(404, "comparison campaign not found")
    return item


@app.get("/api/v2/comparison-campaigns/{campaign_code}/events")
def stream_comparison_campaign_events(campaign_code: str, request: Request):
    _require_trusted(request)
    if RESEARCH.get_comparison_campaign(campaign_code) is None:
        raise HTTPException(404, "comparison campaign not found")

    def stream():
        seen = 0
        while True:
            events = RESEARCH.comparison_campaign_events(campaign_code)
            for index, event in enumerate(events[seen:], start=seen):
                yield (
                    f"id: {index + 1}\n"
                    f"event: {event['type']}\n"
                    f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
                )
            seen = len(events)
            item = RESEARCH.get_comparison_campaign(campaign_code)
            if item is None or item["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v2/research-results", response_model=ResearchResultListResponse)
def list_research_results(
    request: Request,
    runCode: str | None = None,
    truthLabel: Literal[
        "exact_optimal", "robust_best_known", "debatable", "provisional", "rejected"
    ]
    | None = None,
    disposition: Literal["qualified", "rejected", "quarantined"] | None = None,
    minimumScore: float | None = Query(None, ge=0.0, le=100.0),
    search: str | None = Query(None, max_length=200),
    tacticalTag: str | None = Query(None, max_length=100),
    actingSeat: Literal["north", "west", "south", "east"] | None = None,
    observerSeat: Literal["north", "west", "south", "east"] | None = None,
    contract: Literal["sun", "hokum", "ashkal"] | None = None,
    project: str | None = Query(None, max_length=100),
    generationLane: Literal["ortools", "organic_runtime", "exact_input"] | None = None,
    scenarioTemplate: str | None = Query(None, max_length=100),
    scenarioOrigin: Literal["search", "exact_trace_prefix"] | None = None,
    scenarioStatus: ResearchSemanticStatusV1 | None = None,
    requestedProofMode: Literal["exact", "bounded", "policy_conditioned"] | None = None,
    achievedProofMode: Literal["exact", "bounded", "policy_conditioned", "none"]
    | None = None,
    proofCompletion: Literal[
        "exact_complete",
        "bounded_complete",
        "policy_complete",
        "proof_incomplete",
        "proof_failed",
    ]
    | None = None,
    proofVisibility: Literal["complete_state", "actor_information_set", "none"]
    | None = None,
    targetTeam: Literal["north_south", "west_east"] | None = None,
    contractOutcome: str | None = Query(None, max_length=100),
    projectKind: str | None = Query(None, max_length=100),
    projectValue: str | None = Query(None, max_length=20),
    projectState: Literal["eligible", "declared", "comparison_winner", "awarded"]
    | None = None,
    winningRootActionBand: Literal["0", "1", "2_3", "4_plus"] | None = None,
    exactStateAvailable: bool | None = None,
    decisionKind: Literal["bid", "escalation", "project", "play"] | None = None,
    researchBackend: Literal["ortools_cp_sat_v2", "ortools_cp_sat_v3"] | None = None,
    qualificationAuthority: Literal["cp_sat"] | None = None,
    runtimeCompatibilityStatus: Literal["not_checked", "passed", "failed"]
    | None = None,
    configurationBundle: str | None = Query(None, min_length=1, max_length=100),
    exactStateId: str | None = Query(
        None, pattern=r"^STATE-[A-F0-9]{20}$"
    ),
    scenarioCategory: Literal[
        "deal",
        "auction",
        "escalation",
        "projects",
        "playPrefix",
        "state",
        "outcome",
        "tactics",
        "proof",
    ]
    | None = None,
    variantCollection: str | None = Query(
        None, pattern=r"^RUN-[A-Z0-9]{12}$"
    ),
    lessonImpact: list[Annotated[str, Field(json_schema_extra={'enum': list(TAG_AXES['impact'])})]] | None = Query(
        None, min_length=1, max_length=len(TAG_AXES['impact'])),
    lessonTechnique: list[Annotated[str, Field(json_schema_extra={'enum': list(TAG_AXES['technique'])})]] | None = Query(
        None, min_length=1, max_length=len(TAG_AXES['technique'])),
    lessonLevel: list[Annotated[str, Field(json_schema_extra={'enum': list(TAG_AXES['level'])})]] | None = Query(
        None, min_length=1, max_length=len(TAG_AXES['level'])),
    lessonMode: list[Annotated[str, Field(json_schema_extra={'enum': list(TAG_AXES['mode'])})]] | None = Query(
        None, min_length=1, max_length=len(TAG_AXES['mode'])),
    lessonTrickBand: list[Annotated[str, Field(json_schema_extra={'enum': list(TAG_AXES['trickBand'])})]] | None = Query(
        None, min_length=1, max_length=len(TAG_AXES['trickBand'])),
    createdFrom: datetime | None = None,
    createdTo: datetime | None = None,
    groupBy: Literal[
        "tactical_tag",
        "acting_seat",
        "observer_seat",
        "contract",
        "project",
        "generation_lane",
        "truth_label",
        "disposition",
        "scenario_template",
        "scenario_origin",
        "scenario_status",
        "requested_proof_mode",
        "achieved_proof_mode",
        "proof_completion",
        "proof_visibility",
        "target_team",
        "contract_outcome",
        "project_kind",
        "project_value",
        "project_state",
        "winning_root_action_band",
        "exact_state_available",
        "decision_kind",
        "research_backend",
        "qualification_authority",
        "runtime_compatibility_status",
        "configuration_bundle",
        "exact_state_id",
        "scenario_category",
        "variant_collection",
    ]
    | None = None,
    sort: Literal[
        "score_desc",
        "score_asc",
        "confidence_desc",
        "confidence_asc",
        "newest",
        "oldest",
        "result_code_asc",
        "result_code_desc",
    ] = "score_desc",
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    try:
        tag_query = TagQueryV1.build({axis: values for axis, values in (
            ('impact', lessonImpact), ('technique', lessonTechnique), ('level', lessonLevel),
            ('mode', lessonMode), ('trickBand', lessonTrickBand)) if values is not None})
        result = RESEARCH.list_results(
            run_code=runCode,
            truth_label=truthLabel,
            disposition=disposition,
            minimum_score=minimumScore,
            search=search,
            tag_query=tag_query.to_dict(),
            tactical_tag=tacticalTag,
            acting_seat=actingSeat,
            observer_seat=observerSeat,
            contract=contract,
            project=project,
            generation_lane=generationLane,
            **{
                key: value
                for key, value in {
                    "scenario_template": scenarioTemplate,
                    "scenario_origin": scenarioOrigin,
                    "scenario_status": scenarioStatus,
                    "requested_proof_mode": requestedProofMode,
                    "achieved_proof_mode": achievedProofMode,
                    "proof_completion": proofCompletion,
                    "proof_visibility": proofVisibility,
                    "target_team": targetTeam,
                    "contract_outcome": contractOutcome,
                    "project_kind": projectKind,
                    "project_value": projectValue,
                    "project_state": projectState,
                    "winning_root_action_band": winningRootActionBand,
                    "exact_state_available": (
                        None
                        if exactStateAvailable is None
                        else str(exactStateAvailable).lower()
                    ),
                    "decision_kind": decisionKind,
                    "research_backend": researchBackend,
                    "qualification_authority": qualificationAuthority,
                    "runtime_compatibility_status": runtimeCompatibilityStatus,
                    "configuration_bundle": configurationBundle,
                    "exact_state_id": exactStateId,
                    "scenario_category": scenarioCategory,
                    "variant_collection": variantCollection,
                }.items()
                if value is not None
            },
            created_from=createdFrom.isoformat() if createdFrom else None,
            created_to=createdTo.isoformat() if createdTo else None,
            group_by=groupBy,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        return {
            **result,
            "items": [_public_research_result(item) for item in result["items"]],
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post(
    "/api/v2/research-results/{code}:prepare-render",
    response_model=ResearchRenderPrepareResponse,
    response_model_exclude_unset=True,
)
def prepare_research_result_render(
    code: str, body: ResearchRenderPrepareRequest, request: Request
):
    _require_trusted(request)
    try:
        return RESEARCH_RENDERS.prepare(
            code,
            observer_seat=body.observerSeat,
            reveal_policy=body.revealPolicy,
            export_preset=body.exportPreset or "hooked_flagship",
        )
    except KeyError:
        raise HTTPException(404, "research result not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/v2/research-results/{code}", response_model=ResearchResultResponse)
def get_research_result(code: str, request: Request):
    _require_trusted(request)
    item = RESEARCH.get_result(code)
    if item is None:
        raise HTTPException(404, "research result not found")
    return _public_research_result(item)


@app.get(
    "/api/v2/research-results/{result_code}/operator-audit",
    response_model=ResearchOperatorAuditV1Response,
)
def get_research_result_operator_audit(result_code: str, request: Request):
    _require_trusted(request)
    try:
        item = RESEARCH.get_result(result_code)
        if item is None:
            raise HTTPException(404, "research result not found")
        if item.get("resultCode") != result_code:
            raise RuntimeError("stored Research result owner identity mismatch")
        authority = ResearchAuthorityContractV1.from_dict(
            dict(item.get("researchAuthority") or {})
        ).to_dict()
        compatibility = RuntimeCompatibilityContractV1.from_dict(
            dict(item.get("runtimeCompatibility") or {})
        ).to_dict()
        receipt = ExecutionReceiptContractV1.from_dict(
            dict(item.get("executionReceipt") or {})
        ).to_dict()
    except HTTPException:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(
            409, "research result authority validation failed"
        ) from exc
    return {
        "schemaVersion": 1,
        "resultCode": result_code,
        "researchAuthority": authority,
        "runtimeCompatibility": compatibility,
        "executionReceipt": receipt,
    }


@app.get(
    "/api/v2/research-results/{result_code}/quality-audit",
    response_model=ResearchPlayingQualityAuditV1Response,
    response_model_exclude_none=True,
)
def get_research_result_quality_audit(result_code: str, request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.get_playing_quality_audit(result_code)
    except KeyError:
        raise HTTPException(404, "research result not found")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(409, "research playing-quality audit validation failed") from exc


@app.get(
    "/api/v2/research-results/{code}/exact-state",
    response_model=ResearchScenarioExactStateViewV1Response,
)
def get_research_result_exact_state(
    code: str,
    observerSeat: Literal["north", "west", "south", "east"],
    request: Request,
):
    _require_trusted(request)
    try:
        return RESEARCH.get_exact_state(code, observer_seat=observerSeat)
    except KeyError:
        raise HTTPException(404, "research result not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get(
    "/api/v2/research-results/{code}/renders",
    response_model=ResearchResultRenderListResponse,
)
def list_research_result_renders(code: str, request: Request):
    _require_trusted(request)
    if RESEARCH.get_result(code) is None:
        raise HTTPException(404, "research result not found")
    return {
        "resultCode": code,
        "items": RENDER_BATCHES.list_for_source(
            source_kind="research_scenario_result", source_code=code
        ),
    }


@app.post(
    "/api/v2/research-results/{code}:verify-play-compiler",
    response_model=ResearchPlayCompilerVerificationResponse,
    responses={
        404: {
            "model": ResearchMutationErrorResponse,
            "description": "Research result was not found.",
        },
        409: {
            "model": ResearchMutationErrorResponse,
            "description": "Research result cannot be verified.",
        },
        503: {
            "model": ResearchMutationErrorResponse,
            "description": "Play compiler verification is unavailable.",
        },
    },
)
def verify_research_result_play_compiler(code: str, request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.verify_play_compiler(code)
    except KeyError:
        raise HTTPException(404, "research result not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get(
    "/api/v2/research-results/{code}/decisions",
    response_model=ResearchDecisionListResponse,
)
def get_research_result_decisions(code: str, request: Request):
    _require_trusted(request)
    try:
        items = RESEARCH.get_decisions(code)
    except KeyError:
        raise HTTPException(404, "research result not found")
    return {"resultCode": code, "items": items}


@app.get(
    "/api/v2/research-results/{code}/perspectives",
    response_model=ResearchPerspectiveListResponse,
)
def get_research_result_perspectives(
    code: str,
    observerSeat: Literal["north", "west", "south", "east"],
    request: Request,
):
    _require_trusted(request)
    try:
        items = RESEARCH.get_perspectives(code, observerSeat)
    except KeyError:
        raise HTTPException(404, "research result not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {
        "schemaVersion": 1,
        "resultCode": code,
        "observerSeat": observerSeat,
        "items": items,
    }


@app.post(
    "/api/v2/research-results/{code}/decisions/{ordinal}:prepare-render",
    response_model=ResearchRenderPrepareResponse,
    response_model_exclude_unset=True,
)
def prepare_research_decision_render(
    code: str,
    ordinal: int,
    body: ResearchRenderPrepareRequest,
    request: Request,
):
    _require_trusted(request)
    try:
        return RESEARCH_RENDERS.prepare(
            code,
            decision_ordinal=ordinal,
            observer_seat=body.observerSeat,
            reveal_policy=body.revealPolicy,
            export_preset=body.exportPreset or "decision_clip",
        )
    except KeyError:
        raise HTTPException(404, "research result or decision not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get(
    "/api/v2/research-results/{code}/decisions/{ordinal}",
    response_model=ResearchDecisionAuditResponse,
)
def get_research_result_decision(code: str, ordinal: int, request: Request):
    _require_trusted(request)
    try:
        return RESEARCH.get_decision(code, ordinal)
    except KeyError:
        raise HTTPException(404, "research decision audit not found")


@app.get(
    "/api/v2/semantic-disagreements",
    response_model=ResearchSemanticDisagreementListResponse,
)
def list_research_semantic_disagreements(
    request: Request,
    runCode: str | None = None,
    status: Literal["open", "resolved", "dismissed"] | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    return {
        "items": RESEARCH.list_disagreements(
            run_code=runCode, status=status, limit=limit
        )
    }


@app.post("/api/v2/campaigns", status_code=201, response_model=FoundryCampaignResponse)
def create_foundry_campaign(body: FoundryCampaignRequest, request: Request):
    _require_trusted(request)
    try:
        spec = SimulationCampaignSpecV1.build(
            rules_hash=body.rulesHash,
            bundle_hash=body.bundleHash,
            engine_version=body.engineVersion,
            analysis_profile_hash=body.analysisProfileHash,
            perspective_profile_hash=body.perspectiveProfileHash,
            selection_profile_hash=body.selectionProfileHash,
            agent_release=body.agentRelease,
            seat_agents=tuple(
                body.seatAgents[seat] for seat in ("north", "west", "south", "east")
            ),
            lanes=body.lanes,
            lane_profile_hashes=body.laneProfileHashes,
            seed_start=body.seedStart,
            seed_count=body.seedCount,
            seeds_per_shard=body.seedsPerShard,
        )
        return FOUNDRY_CAMPAIGNS.create_campaign(spec)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post(
    "/api/v2/campaigns/{campaign_hash}:execute",
    status_code=202,
    response_model=FoundryCampaignResponse,
)
def execute_foundry_campaign(campaign_hash: str, request: Request):
    _require_trusted(request)
    _require_local_foundry_execution()
    campaign = FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)
    if campaign is None:
        raise HTTPException(404, "campaign not found")
    if campaign["status"] in {"failed", "cancelled"}:
        raise HTTPException(
            409,
            f"campaign cannot execute from {campaign['status']}",
        )
    if campaign["status"] != "completed":
        _schedule_local_foundry_campaign(campaign_hash)
    return FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)


@app.get("/api/v2/foundry-profile", response_model=FoundryRuntimeProfileResponse)
def get_foundry_runtime_profile(request: Request):
    _require_trusted(request)
    return FOUNDRY_PROFILE


@app.get("/api/v2/campaigns", response_model=FoundryCampaignListResponse)
def list_foundry_campaigns(
    request: Request,
    after: str | None = Query(None, pattern=r"^[0-9a-f]{64}$"),
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    values = FOUNDRY_CAMPAIGNS.list_campaigns(after=after, limit=limit)
    return {
        "items": values,
        "nextCursor": values[-1]["campaignHash"] if len(values) == limit else None,
    }


@app.post(
    "/api/v2/analysis-batches",
    status_code=202,
    response_model=FoundryDeepAnalysisJobResponse,
)
def create_foundry_analysis_batch(
    body: FoundryDeepAnalysisCreateRequest,
    request: Request,
):
    _require_trusted(request)
    try:
        return FOUNDRY_DEEP_ANALYSIS.submit(body.campaignHash)
    except KeyError:
        raise HTTPException(404, "campaign not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/analysis-batches/{job_id}:execute",
    status_code=202,
    response_model=FoundryDeepAnalysisJobResponse,
)
def execute_foundry_analysis_batch(job_id: str, request: Request):
    _require_trusted(request)
    _require_local_foundry_execution()
    item = FOUNDRY_DEEP_ANALYSIS.get(job_id)
    if item is None:
        raise HTTPException(404, "deep-analysis job not found")
    if item["status"] in {"failed", "cancelled"}:
        raise HTTPException(
            409,
            f"deep-analysis job cannot execute from {item['status']}",
        )
    if item["status"] != "completed":
        _schedule_local_analysis_worker()
    return FOUNDRY_DEEP_ANALYSIS.get(job_id)


@app.get(
    "/api/v2/analysis-batches",
    response_model=FoundryDeepAnalysisJobListResponse,
)
def list_foundry_analysis_batches(
    request: Request,
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    return {"items": FOUNDRY_DEEP_ANALYSIS.list(status=status, limit=limit)}


@app.post(
    "/api/v2/campaign-equivalence-proofs",
    status_code=201,
    response_model=FoundryCatalogDocumentResponse,
)
def create_foundry_campaign_equivalence_proof(
    body: FoundryCampaignProofCreateRequest,
    request: Request,
):
    _require_trusted(request)
    try:
        return FOUNDRY_CAMPAIGN_PROOFS.create(
            tuple(item.model_dump() for item in body.executions)
        )
    except KeyError:
        raise HTTPException(404, "campaign not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get(
    "/api/v2/analysis-batches/{job_id}",
    response_model=FoundryDeepAnalysisJobResponse,
)
def get_foundry_analysis_batch(job_id: str, request: Request):
    _require_trusted(request)
    item = FOUNDRY_DEEP_ANALYSIS.get(job_id)
    if item is None:
        raise HTTPException(404, "deep-analysis job not found")
    return item


@app.get(
    "/api/v2/analysis-batches/{job_id}/candidates",
    response_model=FoundryAnalysisCandidateListResponse,
)
def list_foundry_analysis_candidates(job_id: str, request: Request):
    _require_trusted(request)
    try:
        return FOUNDRY_ANALYSIS_CANDIDATES.list_candidates(job_id)
    except KeyError:
        raise HTTPException(404, "analysis batch or candidate evidence not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/analysis-batches/{job_id}/candidates/{candidate_hash}:prepare",
    response_model=FoundryAnalysisCandidatePrepareResponse,
)
def prepare_foundry_analysis_candidate(
    job_id: str,
    candidate_hash: str,
    body: FoundryAnalysisCandidatePrepareRequest,
    request: Request,
):
    _require_trusted(request)
    if len(candidate_hash) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_hash
    ):
        raise HTTPException(422, "candidate hash must be a lowercase SHA-256")
    try:
        return FOUNDRY_ANALYSIS_CANDIDATES.prepare_actor_candidate(
            job_id,
            candidate_hash,
            reveal_policy=body.revealPolicy,
            export_preset=body.exportPreset,
        )
    except KeyError:
        raise HTTPException(404, "analysis batch or candidate evidence not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/finalist-validations",
    status_code=202,
    response_model=FoundryFinalistValidationJobResponse,
)
def create_foundry_finalist_validation(
    body: FoundryFinalistValidationCreateRequest,
    request: Request,
):
    _require_trusted(request)
    idempotency_key = request.headers.get("idempotency-key", "").strip()
    if not idempotency_key:
        raise HTTPException(428, "Idempotency-Key header is required")
    if len(idempotency_key) > 200:
        raise HTTPException(422, "Idempotency-Key must not exceed 200 characters")
    try:
        return FOUNDRY_FINALIST_VALIDATIONS.submit(
            analysis_job_id=body.analysisJobId,
            candidate_hashes=tuple(body.candidateHashes),
            idempotency_key=idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get(
    "/api/v2/finalist-validations",
    response_model=FoundryFinalistValidationJobListResponse,
)
def list_foundry_finalist_validations(
    request: Request,
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    return {"items": FOUNDRY_FINALIST_VALIDATIONS.list(status=status, limit=limit)}


@app.get(
    "/api/v2/finalist-validations/{job_id}",
    response_model=FoundryFinalistValidationJobResponse,
)
def get_foundry_finalist_validation(job_id: str, request: Request):
    _require_trusted(request)
    item = FOUNDRY_FINALIST_VALIDATIONS.get(job_id)
    if item is None:
        raise HTTPException(404, "finalist validation job not found")
    return item


@app.post(
    "/api/v2/finalist-validations/{job_id}:prepare-best",
    response_model=FoundryAnalysisCandidatePrepareResponse,
)
def prepare_best_foundry_finalist_candidate(
    job_id: str,
    body: FoundryAnalysisCandidatePrepareRequest,
    request: Request,
):
    _require_trusted(request)
    item = FOUNDRY_FINALIST_VALIDATIONS.get(job_id)
    if item is None:
        raise HTTPException(404, "finalist validation job not found")
    try:
        return FOUNDRY_ANALYSIS_CANDIDATES.prepare_best_finalist_candidate(
            item,
            reveal_policy=body.revealPolicy,
            export_preset=body.exportPreset,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/finalist-validations/{job_id}:cancel",
    response_model=FoundryFinalistValidationJobResponse,
)
def cancel_foundry_finalist_validation(job_id: str, request: Request):
    _require_trusted(request)
    existing = FOUNDRY_FINALIST_VALIDATIONS.get(job_id)
    if existing is None:
        raise HTTPException(404, "finalist validation job not found")
    if not FOUNDRY_FINALIST_VALIDATIONS.cancel(job_id):
        raise HTTPException(
            409,
            f"finalist validation cannot be cancelled from {existing['status']}",
        )
    return FOUNDRY_FINALIST_VALIDATIONS.get(job_id)


@app.post(
    "/api/v2/analysis-batches/{job_id}:cancel",
    response_model=FoundryDeepAnalysisJobResponse,
)
def cancel_foundry_analysis_batch(job_id: str, request: Request):
    _require_trusted(request)
    if not FOUNDRY_DEEP_ANALYSIS.cancel(job_id):
        item = FOUNDRY_DEEP_ANALYSIS.get(job_id)
        if item is None:
            raise HTTPException(404, "deep-analysis job not found")
        raise HTTPException(
            409, f"deep-analysis job cannot be cancelled from {item['status']}"
        )
    return FOUNDRY_DEEP_ANALYSIS.get(job_id)


@app.post(
    "/api/v2/render-batches",
    status_code=202,
    response_model=FoundryRenderBatchJobResponse,
    responses={
        422: {
            "model": ResearchMutationValidationErrorResponse,
            "description": "Render batch request or Idempotency-Key is invalid.",
        },
        428: {
            "model": ResearchMutationErrorResponse,
            "description": "Idempotency-Key is required.",
        },
    },
    openapi_extra=_RENDER_BATCH_MUTATION_OPENAPI,
)
def create_foundry_render_batch(
    body: FoundryRenderBatchCreateRequest,
    request: Request,
):
    _require_trusted(request)
    idempotency_key = _require_trimmed_idempotency_key(
        request,
        maximum_length=200,
    )
    try:
        return RENDER_BATCHES.submit(
            tuple(
                item.model_dump(mode="json", exclude_unset=True) for item in body.items
            ),
            idempotency_key=idempotency_key,
            playback_evidence=body.playbackEvidence,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get(
    "/api/v2/render-batches",
    response_model=FoundryRenderBatchJobListResponse,
)
def list_foundry_render_batches(
    request: Request,
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    return {"items": RENDER_BATCHES.list(status=status, limit=limit)}


@app.get(
    "/api/v2/render-batches/{job_id}",
    response_model=FoundryRenderBatchJobResponse,
)
def get_foundry_render_batch(job_id: str, request: Request):
    _require_trusted(request)
    item = RENDER_BATCHES.get(job_id)
    if item is None:
        raise HTTPException(404, "render batch job not found")
    return item


@app.post(
    "/api/v2/render-batches/{job_id}:cancel",
    response_model=FoundryRenderBatchJobResponse,
)
def cancel_foundry_render_batch(job_id: str, request: Request):
    _require_trusted(request)
    existing = RENDER_BATCHES.get(job_id)
    if existing is None:
        raise HTTPException(404, "render batch job not found")
    if not RENDER_BATCHES.cancel(job_id):
        raise HTTPException(
            409, f"render batch job cannot be cancelled from {existing['status']}"
        )
    return RENDER_BATCHES.get(job_id)


@app.get("/api/v2/render-batches/{job_id}/events")
def stream_foundry_render_batch_events(job_id: str, request: Request):
    _require_trusted(request)
    if RENDER_BATCHES.get(job_id) is None:
        raise HTTPException(404, "render batch job not found")

    def stream():
        seen = 0
        while True:
            events = RENDER_BATCHES.events(job_id) or []
            for index, event in enumerate(events[seen:], start=seen):
                yield (
                    f"id: {index + 1}\n"
                    f"event: {event['type']}\n"
                    f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
                )
            seen = len(events)
            item = RENDER_BATCHES.get(job_id)
            if item is None or item["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/api/v2/trace-import-catalogs",
    status_code=201,
    response_model=FoundryCatalogDocumentResponse,
)
def create_trace_import_catalog(
    body: TraceImportCatalogCreateRequest, request: Request
):
    _require_trusted(request)
    try:
        return FOUNDRY_CATALOGS.create_trace_import_catalog(
            body.lane,
            tuple(item.model_dump() for item in body.entries),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/score-reachability-catalogs",
    status_code=201,
    response_model=FoundryCatalogDocumentResponse,
)
def create_score_reachability_catalog(
    body: ScoreReachabilityCatalogCreateRequest,
    request: Request,
):
    _require_trusted(request)
    try:
        return FOUNDRY_CATALOGS.create_score_reachability_catalog(
            tuple(item.model_dump() for item in body.traceReferences)
        )
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, "complete-trace artifact not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post(
    "/api/v2/score-reachability-catalogs/{catalog_hash}:prove",
    status_code=201,
    response_model=FoundryCatalogDocumentResponse,
)
def prove_score_reachability(
    catalog_hash: str,
    body: ScoreReachabilityProveRequest,
    request: Request,
):
    _require_trusted(request)
    try:
        return FOUNDRY_CATALOGS.prove_score_context(
            catalog_hash,
            body.targetScore,
            target_dealer=body.targetDealer,
            initial_dealer=body.initialDealer,
        )
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, "score reachability catalog not found")
    except LookupError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get(
    "/api/v2/foundry-catalogs",
    response_model=FoundryCatalogListResponse,
)
def list_foundry_catalogs(
    request: Request,
    kind: Literal[
        "trace_import_catalog_v1",
        "score_reachability_catalog_v1",
        "score_reachability_proof_v1",
        "campaign_equivalence_proof_v1",
        "natural_sakka_generation_v1",
        "qualified_laffa_v1",
        "qualified_laffa_v2",
    ]
    | None = None,
    after: str | None = Query(None, pattern=r"^[0-9a-f]{64}$"),
    limit: int = Query(100, ge=1, le=500),
):
    _require_trusted(request)
    try:
        return FOUNDRY_CATALOGS.list(kind=kind, after=after, limit=limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get(
    "/api/v2/foundry-catalogs/{catalog_hash}",
    response_model=FoundryCatalogDocumentResponse,
)
def get_foundry_catalog(catalog_hash: str, request: Request):
    _require_trusted(request)
    try:
        return FOUNDRY_CATALOGS.get(catalog_hash)
    except KeyError:
        raise HTTPException(404, "Foundry catalog not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/v2/campaigns/{campaign_hash}", response_model=FoundryCampaignResponse)
def get_foundry_campaign(campaign_hash: str, request: Request):
    _require_trusted(request)
    item = FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)
    if item is None:
        raise HTTPException(404, "campaign not found")
    return item


@app.get(
    "/api/v2/campaigns/{campaign_hash}/shards", response_model=FoundryShardListResponse
)
def list_foundry_shards(campaign_hash: str, request: Request):
    _require_trusted(request)
    if FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash) is None:
        raise HTTPException(404, "campaign not found")
    return {
        "campaignHash": campaign_hash,
        "shards": FOUNDRY_CAMPAIGNS.list_shards(campaign_hash),
    }


@app.get(
    "/api/v2/campaigns/{campaign_hash}/attempts",
    response_model=FoundryShardAttemptListResponse,
)
def list_foundry_campaign_attempts(campaign_hash: str, request: Request):
    _require_trusted(request)
    if FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash) is None:
        raise HTTPException(404, "campaign not found")
    return {
        "campaignHash": campaign_hash,
        "attempts": FOUNDRY_CAMPAIGNS.list_attempts(campaign_hash),
    }


@app.post(
    "/api/v2/campaigns/{campaign_hash}:cancel", response_model=FoundryCampaignResponse
)
def cancel_foundry_campaign(campaign_hash: str, request: Request):
    _require_trusted(request)
    if not FOUNDRY_CAMPAIGNS.request_cancel(campaign_hash):
        item = FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)
        if item is None:
            raise HTTPException(404, "campaign not found")
        raise HTTPException(409, f"campaign cannot be cancelled from {item['status']}")
    return FOUNDRY_CAMPAIGNS.get_campaign(campaign_hash)


@app.post("/api/v2/shards:claim", response_model=FoundryShardClaimResponse)
def claim_foundry_shard(body: FoundryShardClaimRequest, request: Request):
    _require_trusted(request)
    claimed = FOUNDRY_CAMPAIGNS.claim_shard(
        body.workerId,
        lease_seconds=body.leaseSeconds,
        campaign_hash=body.campaignHash,
        supported_lanes=tuple(body.supportedLanes),
    )
    return {"claim": claimed.to_dict() if claimed is not None else None}


@app.post("/api/v2/shards/{spec_hash}:heartbeat")
def heartbeat_foundry_shard(
    spec_hash: str, body: FoundryShardHeartbeatRequest, request: Request
):
    _require_trusted(request)
    accepted = FOUNDRY_CAMPAIGNS.heartbeat(
        spec_hash,
        worker_id=body.workerId,
        fencing_token=body.fencingToken,
        lease_seconds=body.leaseSeconds,
    )
    if not accepted:
        raise HTTPException(409, "stale, expired, or unknown shard claim")
    return {"accepted": True, "specHash": spec_hash}


@app.post("/api/v2/shards/{spec_hash}:commit", response_model=FoundryShardResponse)
def commit_foundry_shard(
    spec_hash: str, body: FoundryShardCommitRequest, request: Request
):
    _require_trusted(request)
    row = FOUNDRY_CAMPAIGNS.get_shard(spec_hash)
    if row is None:
        raise HTTPException(404, "shard not found")
    try:
        spec = SimulationShardSpecV1.from_dict(row["spec"])
        result = SimulationShardResultV1.build(
            spec,
            processed_seeds=body.processedSeeds,
            trace_hashes=body.traceHashes,
            candidate_hashes=body.candidateHashes,
            coverage_cells=body.coverageCells,
            artifact_references=(item.model_dump() for item in body.artifactReferences),
            completion_status=body.completionStatus,
        )
        return FOUNDRY_CAMPAIGNS.commit_result(
            result,
            worker_id=body.workerId,
            fencing_token=body.fencingToken,
        )
    except PermissionError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/v1/challenges")
def list_challenges(status: str | None = None, limit: int = 100):
    return {"challenges": LEARNING_STORE.list_challenges(status=status, limit=limit)}


@app.post(
    "/api/v2/laffat/{identifier}:prepare-render",
    response_model=(
        ResearchRenderPrepareResponse
        | LaffaRenderPrepareResponse
        | ResearchAuditTraceRenderPrepareResponse
    ),
    response_model_exclude_unset=True,
)
def prepare_any_laffa_render(
    identifier: str,
    body: ResearchRenderPrepareRequest,
    request: Request,
):
    """Prepare any complete replay-valid generated Laffa as non-publishable media."""

    _require_trusted(request)
    try:
        if RESEARCH.get_result(identifier) is not None:
            return RESEARCH_RENDERS.prepare(
                identifier,
                observer_seat=body.observerSeat,
                reveal_policy=body.revealPolicy,
                export_preset=body.exportPreset,
            )
        audit_trace = LAFFA_LIBRARY.resolve_audit_trace(identifier)
        if audit_trace is not None:
            return RESEARCH_RENDERS.prepare_audit_trace(
                audit_trace,
                observer_seat=body.observerSeat,
                reveal_policy=body.revealPolicy,
                export_preset=body.exportPreset,
            )
        sakka_round = LAFFA_LIBRARY.resolve_sakka_round(identifier)
        if sakka_round is not None:
            return RESEARCH_RENDERS.prepare_trace_reference(
                dict(sakka_round["traceReference"]),
                code=str(sakka_round["code"]),
                observer_seat=body.observerSeat,
                reveal_policy=body.revealPolicy,
                export_preset=body.exportPreset,
                source_kind="sakka_round",
            )
        return RESEARCH_RENDERS.prepare_laffa(
            identifier,
            observer_seat=body.observerSeat,
            reveal_policy=body.revealPolicy,
            export_preset=body.exportPreset,
        )
    except KeyError:
        raise HTTPException(404, "laffa not found")
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v2/laffat/{identifier}/renders")
def list_any_laffa_renders(identifier: str, request: Request):
    _require_trusted(request)
    if RESEARCH.get_result(identifier) is not None:
        jobs = {
            job["jobId"]: job
            for source_kind in ("research_result", "research_scenario_result")
            for job in RENDER_BATCHES.list_for_source(
                source_kind=source_kind, source_code=identifier
            )
        }
        return {
            "laffaCode": identifier,
            "items": list(jobs.values()),
        }
    audit_trace = LAFFA_LIBRARY.resolve_audit_trace(identifier)
    if audit_trace is not None:
        code = str(audit_trace["auditTraceCode"])
        return {
            "laffaCode": code,
            "items": RENDER_BATCHES.list_for_source(
                source_kind="laffa", source_code=code
            ),
        }
    sakka_round = LAFFA_LIBRARY.resolve_sakka_round(identifier)
    if sakka_round is not None:
        code = str(sakka_round["code"])
        return {
            "laffaCode": code,
            "items": RENDER_BATCHES.list_for_source(
                source_kind="laffa", source_code=code
            ),
        }
    game = LEARNING_STORE.get_game(identifier)
    if game is None:
        raise HTTPException(404, "laffa not found")
    code = str(game.get("serial") or game.get("gameHash") or identifier)
    return {
        "laffaCode": code,
        "items": RENDER_BATCHES.list_for_source(source_kind="laffa", source_code=code),
    }


@app.get("/api/v2/laffat/{identifier}/decisions")
def get_laffa_decision_ledger(identifier: str, request: Request):
    """Return the complete actor-safe decision index for one recorded لفة."""
    _require_trusted(request)
    try:
        ledger = LEARNING_STORE.decision_ledger(identifier)
    except KeyError:
        raise HTTPException(404, "laffa not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    counts = {seat: 0 for seat in ("north", "west", "south", "east")}
    for item in ledger.records:
        counts[item.acting_seat.name.lower()] += 1
    return {**ledger.to_dict(), "decisionCountsBySeat": counts}


@app.get("/api/v2/laffat/{identifier}/complete-trace")
def get_laffa_complete_trace(identifier: str, request: Request):
    """Internal trusted trace. It is never exposed through a public locator."""
    _require_trusted(request)
    try:
        trace = LEARNING_STORE.complete_game_trace(identifier)
    except KeyError:
        raise HTTPException(404, "laffa not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return trace.to_dict(trusted=True)


@app.get("/api/v2/laffat/{identifier}/perspectives")
def get_laffa_observer_timeline(identifier: str, observer: str, request: Request):
    """Return one isolated observer timeline; combined private views are forbidden."""
    _require_trusted(request)
    try:
        seat = Seat.parse(observer)
        game = LEARNING_STORE.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        decisions = LEARNING_STORE.list_decisions(identifier, limit=10000)
        from baloot_platform.challenge.perspective_pipeline import (
            observer_projection_timeline,
        )

        projections = observer_projection_timeline(LEARNING_STORE, decisions, seat)
    except KeyError:
        raise HTTPException(404, "laffa not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {
        "schemaVersion": 1,
        "laffaId": str(game.get("gameHash") or game.get("gameId") or identifier),
        "observerSeat": seat.name.lower(),
        "projections": [item.to_dict() for item in projections],
    }


def _foundry_laffa_evidence(identifier: str):
    audits = LEARNING_STORE.list_game_audits(identifier, limit=1)
    if not audits:
        raise HTTPException(409, "laffa requires a complete observer-safe audit first")
    plugin = REGISTRY.get("perspective_analyzer", "multi_perspective_foundry_v1")
    try:
        return plugin.analyze_laffa(identifier, audits[0])
    except KeyError:
        raise HTTPException(404, "laffa not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/v2/laffat/{identifier}/evidence")
def get_laffa_evidence(identifier: str, request: Request):
    _require_trusted(request)
    return _foundry_laffa_evidence(identifier).to_dict()


@app.get("/api/v2/laffat/{identifier}/scenario-graph")
def get_laffa_scenario_graph(identifier: str, request: Request):
    _require_trusted(request)
    evidence = _foundry_laffa_evidence(identifier)
    if evidence.scenario_graph is None:
        raise HTTPException(409, "scenario graph has not been generated")
    return evidence.scenario_graph.to_dict()


@app.post("/api/v2/challenge-locators", status_code=201)
def create_foundry_challenge_locator(
    body: FoundryChallengeLocatorCreateRequest,
    request: Request,
):
    """Create an opaque public link bound to one exact observer-safe state."""
    _require_trusted(request)
    try:
        trace = LEARNING_STORE.complete_game_trace(body.laffaIdentifier)
        if body.analysisArtifact is not None:
            if body.promptContract != "choose_acting_action":
                raise ValueError(
                    "governed analysis artifacts apply only to actor challenges"
                )
            record = next(
                (
                    item
                    for item in trace.decision_ledger.records
                    if item.decision_id == body.decisionId
                ),
                None,
            )
            if record is None:
                raise KeyError(body.decisionId)
            if body.observerSeat != record.acting_seat.name.lower():
                raise ValueError(
                    "governed actor challenge observer must be the acting seat"
                )
            published = CHALLENGE_PUBLICATION.publish_actor_challenge(
                trace,
                analysis_reference=body.analysisArtifact.model_dump(),
                decision_id=body.decisionId,
                reveal_policy=body.revealPolicy,
                expires_at=body.expiresAt,
            )
            locator = published.locator
        else:
            locator = CHALLENGE_LOCATORS.publish(
                trace,
                decision_id=body.decisionId,
                observer_seat=body.observerSeat,
                prompt_contract=body.promptContract,
                reveal_policy=body.revealPolicy,
                expires_at=body.expiresAt,
            )
        return locator.to_dict()
    except KeyError:
        raise HTTPException(404, "laffa or decision not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


def _ensure_research_challenge_locator_v1(code: str) -> bool:
    normalized_code = str(code).upper()
    if re.fullmatch(r"LAF-[A-F0-9]{10}", normalized_code) is None:
        return False
    result = RESEARCH.get_result(normalized_code)
    if result is None or result.get("resultCode") != normalized_code:
        raise KeyError(code)
    qualification = dict(result.get("qualification") or {})
    proof = dict(result.get("proofOutcome") or {})
    projection = dict(result.get("scenarioProjection") or {})
    compatibility = RuntimeCompatibilityContractV1.from_dict(
        dict(result.get("runtimeCompatibility") or {})
    ).to_dict()
    identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
    exact_state_id = str(projection.get("exactStateId") or "")
    actors = [str(item) for item in result.get("interestingSeats") or ()]
    if (
        result.get("schemaVersion") != 2
        or result.get("classification") != "scenario_v4"
        or result.get("disposition") != "qualified"
        or qualification.get("qualified") is not True
        or qualification.get("observerSafe") is not True
        or qualification.get("decisionPointAuditComplete") is not True
        or list(qualification.get("failures") or ())
        or proof.get("claimSatisfied") is not True
        or proof.get("completionStatus")
        not in {"exact_complete", "bounded_complete", "policy_complete"}
        or compatibility.get("status") != "passed"
        or projection.get("exactStateAvailable") is not True
        or re.fullmatch(r"STATE-[A-F0-9]{20}", exact_state_id) is None
        or identity.get("exactStateId") != exact_state_id
        or len(actors) != 1
    ):
        raise ValueError("Research result is not playable")
    public_exact_state = RESEARCH.get_exact_state(
        normalized_code,
        observer_seat=actors[0],
    )
    CHALLENGE_LOCATORS.publish_research_result(
        result_code=normalized_code,
        result_hash=str(result.get("resultHash") or ""),
        exact_state_id=exact_state_id,
        public_exact_state=public_exact_state,
        proof_outcome=proof,
        truth_label=str(result.get("truthLabel") or ""),
        confidence=float(result.get("confidence", 0.0)),
    )
    return True


@app.get("/api/v2/challenge-locators/{code}")
def resolve_foundry_challenge_locator(code: str):
    """Resolve public pre-attempt state without recommendation or hidden IDs."""
    try:
        _ensure_research_challenge_locator_v1(code)
        return CHALLENGE_LOCATORS.resolve(code)
    except KeyError:
        raise HTTPException(404, "challenge code not found")
    except PermissionError as exc:
        raise HTTPException(410, str(exc))
    except (RuntimeError, TypeError, ValueError):
        raise HTTPException(404, "challenge code not found")


@app.get("/api/v2/challenge-locators/{code}/attempt")
def restore_foundry_challenge_attempt(code: str, request: Request):
    """Restore this session's committed reveal without accepting another choice."""
    try:
        identity = getattr(request.state, "identity", None)
        access_claims = getattr(request.state, "access_claims", None)
        authenticated_subject = str(
            getattr(access_claims, "subject_id", "")
            or getattr(identity, "subject_id", "")
            or ""
        )
        if not authenticated_subject:
            return {"schemaVersion": 1, "committed": False}
        try:
            _ensure_research_challenge_locator_v1(code)
        except (KeyError, RuntimeError, TypeError, ValueError):
            raise HTTPException(404, "challenge code not found")
        restored = CHALLENGE_LOCATORS.restore_attempt(
            code,
            participant_id=authenticated_subject,
            include_friend_statistics=True,
        )
        return restored or {"schemaVersion": 1, "committed": False}
    except KeyError:
        raise HTTPException(404, "challenge code not found")
    except PermissionError as exc:
        raise HTTPException(410, str(exc))


@app.post("/api/v2/challenge-locators/{code}/attempts", status_code=201)
def commit_foundry_challenge_attempt(
    code: str,
    body: FoundryChallengeAttemptRequest,
    request: Request,
):
    """Commit once, then disclose perspective-specific choice statistics."""
    try:
        identity = getattr(request.state, "identity", None)
        access_claims = getattr(request.state, "access_claims", None)
        authenticated_subject = str(
            getattr(access_claims, "subject_id", "")
            or getattr(identity, "subject_id", "")
            or ""
        )
        if not authenticated_subject:
            raise HTTPException(
                401,
                "create a guest or account session before committing a challenge",
            )
        if body.participantId and body.participantId != authenticated_subject:
            raise HTTPException(
                403,
                "authenticated challenge attempts must use the session subject",
            )
        try:
            _ensure_research_challenge_locator_v1(code)
        except (KeyError, RuntimeError, TypeError, ValueError):
            raise HTTPException(404, "challenge code not found")
        return CHALLENGE_LOCATORS.commit_attempt(
            code,
            participant_id=authenticated_subject,
            chosen_action=body.chosenAction,
            response_seconds=body.responseSeconds,
            confidence=body.confidence,
            changed_answer=body.changedAnswer,
            include_friend_statistics=bool(authenticated_subject),
            calibration_eligible=bool(authenticated_subject),
        )
    except KeyError:
        raise HTTPException(404, "challenge code not found")
    except PermissionError as exc:
        raise HTTPException(410, str(exc))
    except ChallengeAttemptConflictError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/v2/challenge-locators/{code}:revoke")
def revoke_foundry_challenge_locator(code: str, request: Request):
    _require_trusted(request)
    if not CHALLENGE_LOCATORS.revoke(code):
        raise HTTPException(404, "challenge code not found or already revoked")
    return {"code": code.upper(), "status": "revoked"}


@app.get("/api/v1/challenges/{challenge_id}")
def get_challenge(challenge_id: str):
    item = LEARNING_STORE.get_challenge(challenge_id)
    if item is None:
        raise HTTPException(404, "challenge not found")
    return item


@app.post("/api/v1/challenges/{challenge_id}:status")
def set_challenge_status(
    challenge_id: str, body: ChallengeStatusRequest, request: Request
):
    try:
        identity, _ = require_identity(request, IDENTITY)
        validate_reviewer_role(identity.kind, _CHALLENGE_PUBLICATION_POLICY)
        return LEARNING_STORE.update_challenge_status(challenge_id, body.status)
    except KeyError:
        raise HTTPException(404, "challenge not found")
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/v1/challenges/{challenge_id}/publication-reviews")
def list_challenge_publication_reviews(challenge_id: str, request: Request):
    require_identity(request, IDENTITY)
    if LEARNING_STORE.get_challenge(challenge_id) is None:
        raise HTTPException(404, "challenge not found")
    return {
        "reviews": LEARNING_STORE.list_challenge_reviews(challenge_id),
        "events": LEARNING_STORE.list_challenge_publication_events(challenge_id),
    }


@app.post("/api/v1/challenges/{challenge_id}/publication-reviews", status_code=201)
def record_challenge_publication_review(
    challenge_id: str, body: ChallengePublicationReviewRequest, request: Request
):
    identity, _ = require_identity(request, IDENTITY)
    try:
        return LEARNING_STORE.record_challenge_review(
            challenge_id,
            reviewer_id=identity.subject_id,
            reviewer_role=identity.kind,
            decision=body.decision,
            evidence_sha256=body.evidenceSha256,
            comment=body.comment,
            policy=_CHALLENGE_PUBLICATION_POLICY,
        )
    except KeyError:
        raise HTTPException(404, "challenge not found")
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _govern_challenge(
    challenge_id: str, target: str, body: ChallengeGovernanceRequest, request: Request
):
    identity, _ = require_identity(request, IDENTITY)
    try:
        return LEARNING_STORE.govern_challenge_status(
            challenge_id,
            target=target,
            actor_id=identity.subject_id,
            actor_role=identity.kind,
            evidence_sha256=body.evidenceSha256,
            policy=_CHALLENGE_PUBLICATION_POLICY,
        )
    except KeyError:
        raise HTTPException(404, "challenge not found")
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/v1/challenges/{challenge_id}:approve")
def approve_challenge(
    challenge_id: str, body: ChallengeGovernanceRequest, request: Request
):
    return _govern_challenge(challenge_id, "approved", body, request)


@app.post("/api/v1/challenges/{challenge_id}:publish")
def publish_challenge(
    challenge_id: str, body: ChallengeGovernanceRequest, request: Request
):
    return _govern_challenge(challenge_id, "published", body, request)


@app.post("/api/v1/challenges/{challenge_id}:retire")
def retire_challenge(
    challenge_id: str, body: ChallengeGovernanceRequest, request: Request
):
    return _govern_challenge(challenge_id, "retired", body, request)


@app.post("/api/v1/challenges/{challenge_id}:render", status_code=202)
def render_challenge(challenge_id: str, request: ChallengeRenderRequest):
    if LEARNING_STORE.get_challenge(challenge_id) is None:
        raise HTTPException(404, "challenge not found")
    job_id = uuid.uuid4().hex[:16]
    STORE.create(
        job_id,
        {"challengeId": challenge_id, **request.model_dump()},
        kind="challenge_video",
    )
    EXECUTOR.submit(
        _execute_challenge_render_job, job_id, challenge_id, request.model_dump()
    )
    return {"jobId": job_id, "challengeId": challenge_id, "status": "queued"}


def main() -> None:
    uvicorn.run(
        "baloot_platform.service.api:app",
        host=os.environ.get("BALOOT_HOST", "0.0.0.0"),
        port=int(os.environ.get("BALOOT_PORT", "8080")),
        reload=False,
    )
