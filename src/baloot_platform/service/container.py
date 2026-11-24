"""Application dependency container.

The engine and domain libraries remain independent from databases and HTTP.
This container assembles replaceable production adapters once, making tests,
workers and future service extraction explicit instead of duplicating global
construction logic throughout the API module.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.agents.qualification import (
    FinalistAgentPilotEvidenceV1,
    FoundryAgentQualificationBundleV1,
    FoundryAgentReleaseV1,
    validate_finalist_analysis_compatibility,
)
from baloot_platform.adjudication import (
    AdjudicationProfile,
    AdjudicationService,
    PostgresAdjudicationStore,
    SQLiteAdjudicationStore,
)
from baloot_platform.analysis.search_backend import (
    NativeExactSearchBackendV3,
    PythonReferenceSearchBackend,
)
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.evaluator import AnalysisPolicy
from baloot_platform.audit import OnlineDifficultyCalibrator, ReplayAuditService
from baloot_platform.challenge.miner import ProgressiveChallengeMiner
from baloot_platform.challenge.foundry import MultiPerspectiveFoundryV1
from baloot_platform.challenge.tactical_evidence import (
    CausalTacticalDetectorV1,
    TacticalDetectorProfileV1,
)
from baloot_platform.challenge.campaign_control import (
    PostgresCampaignControlStore,
    SQLiteCampaignControlStore,
)
from baloot_platform.challenge.executor import DeterministicBatchExecutorV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.candidate_resimulation import (
    CandidateResimulationProfileV1,
    CandidateResimulationV1,
)
from baloot_platform.challenge.finalist_validation import (
    FinalistLaffaReplayV1,
    FinalistValidationProfileV1,
)
from baloot_platform.challenge.finalist_validation_service import (
    FinalistValidationJobServiceV1,
)
from baloot_platform.challenge.finalist_adjudication import (
    AdaptiveFinalistTraceAnalyzerV1,
    FinalistSurroundingAdjudicationProfileV1,
)
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.analysis_pipeline import (
    CascadedTraceAnalysisProfileV1,
    CascadedTraceAnalyzerV1,
    DecisionAnalysisExecutorProfileV1,
    StagedTraceAnalyzerV1,
    TraceAnalysisProfileV1,
    build_finalist_reanalysis_trace_profile_v1,
    trace_analysis_composite_identity_hash_v1,
)
from baloot_platform.challenge.funnel import FoundryAnalysisFunnelV1
from baloot_platform.challenge.profiles import (
    PerspectivePolicyV1,
    SelectionProfileV1,
    ThrillProfileV1,
)
from baloot_platform.challenge.scoring import ProfileChallengeScorerV1
from baloot_platform.challenge.quality_diversity import (
    QualityDiversityProfileV1,
    QualityDiversitySelectorV1,
)
from baloot_platform.challenge.deep_analysis import (
    DeepAnalysisProfileV1,
    FoundryDeepAnalysisServiceV1,
)
from baloot_platform.challenge.video_production_batch import HookedLaffaBatchProfileV1
from baloot_platform.challenge.campaign_proof import (
    CampaignProofProfileV1,
    CampaignProofServiceV1,
)
from baloot_platform.challenge.screening import OrganicScreeningProfileV1
from baloot_platform.challenge.guided_search import (
    GuidedLegalSearchV1,
    GuidedSearchProfileV1,
)
from baloot_platform.challenge.constraint_deal_search import OrToolsCpSatDealGeneratorV1
from baloot_platform.challenge.disagreement_search import (
    DisagreementSearchProfileV1,
    DisagreementSearchV1,
)
from baloot_platform.challenge.context_search import (
    CriticalContextSearchProfileV1,
    CriticalContextSearchV1,
    NaturalHistoryContextVerifierV1,
)
from baloot_platform.challenge.context_search_service import ContextSearchJobServiceV1
from baloot_platform.challenge.sakka_context import CuratedSakkaSignerV1
from baloot_platform.challenge.sakka_generation import (
    NaturalSakkaGenerationProfileV1,
    NaturalSakkaGeneratorV1,
)
from baloot_platform.challenge.sakka_generation_service import (
    NaturalSakkaGenerationJobServiceV1,
)
from baloot_platform.challenge.publication import ChallengeAnswerPolicyV1
from baloot_platform.challenge.publication_service import ChallengePublicationServiceV1
from baloot_platform.challenge.analysis_catalog import FoundryAnalysisCandidateServiceV1
from baloot_platform.challenge.trace_import import (
    TraceImportCatalogFactoryV1,
    TraceImportPolicyV1,
)
from baloot_platform.challenge.score_reachability import (
    ScoreReachabilityCatalogFactoryV1,
    ScoreReachabilityProfileV1,
)
from baloot_platform.challenge.catalog_control import (
    FoundryCatalogServiceV1,
    PostgresFoundryCatalogStoreV1,
    SQLiteFoundryCatalogStoreV1,
)
from baloot_platform.challenge.oracle import ActualDealOracleVerifierV1
from baloot_platform.challenge.locator import (
    ChallengeLocatorServiceV1,
    PostgresChallengeLocatorStore,
    SQLiteChallengeLocatorStore,
)
from baloot_platform.config import (
    ProfileBundle,
    build_bundle,
    data_root,
    platform_root,
    profile_bundle_from_resolved,
)
from baloot_platform.configuration import (
    PostgresConfigurationStore,
    SQLiteConfigurationStore,
    VersionedConfigurationService,
)
from baloot_platform.configuration.schedule import ConfigurationIntegrityError
from baloot_platform.identity import (
    IdentityService,
    PostgresIdentityStore,
    SQLiteIdentityStore,
)
from baloot_platform.history.canonical_store_v17 import CanonicalHistoryStore
from baloot_platform.learning.service import LearningService
from baloot_platform.explain import ExplanationService
from baloot_platform.review import DecisionReviewService
from baloot_platform.research import (
    BalootDomainCompilerV1,
    BalootDomainSpecV2,
    CompositeResearchAuditTraceRepositoryV1,
    OrToolsCpSatAuctionCompilerV1,
    OrToolsCpSatEscalationCompilerV1,
    OrToolsCpSatProjectCompilerV1,
    OrToolsCpSatPlayCompilerV1,
    PostgresResearchStoreV1,
    ResearchExecutionContextFactoryV1,
    ResearchFoundryServiceV1,
    ResearchAuditTraceRepositoryV1,
    ResearchRenderPreparationServiceV1,
    SQLiteResearchStoreV1,
)
from baloot_platform.hitl import (
    SQLiteReviewGovernanceStore,
    PostgresReviewGovernanceStore,
    ReviewGovernanceServiceV18,
)
from baloot_platform.learning.store import LearningStore
from baloot_platform.live import (
    InMemoryLiveEventBroker,
    LiveTableService,
    PostgresLiveStore,
    RedisLiveEventBroker,
    SQLiteLiveStore,
)
from baloot_platform.plugins.platform import (
    GameRepositoryAdapter,
    HistoryStoreAdapter,
    LearningBackendAdapter,
)
from baloot_platform.performance import PerformanceService
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable
from baloot_platform.plugins.registry import PluginPolicy, PluginRegistry
from baloot_platform.sakka import PostgresSakkaStore, SakkaService, SQLiteSakkaStore
from baloot_platform.sakka.signing import HmacBoardSigner
from baloot_platform.security import (
    AesGcmPayloadCipher,
    AuthorizationPolicy,
    OidcCallbackService,
    OidcProviderConfig,
    PasskeyChallengeService,
    PkceTransactionStore,
    PostgresSecurityStateStore,
    SQLiteSecurityStateStore,
    SignedAccessTokenService,
    WebSocketTicketService,
)
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore
from baloot_platform.storage.jobs import JobStore
from baloot_platform.storage.postgres import PostgresGameStore, PostgresJobStore
from baloot_platform.storage.postgres_learning import PostgresLearningStore
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
)
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.batch_executor import RenderBatchExecutorV1
from baloot_platform.video.batch_service import RenderBatchServiceV1
from baloot_platform.video.multi_angle_composer import MultiAngleComposerV1
from baloot_platform.video.hitl_review_v18 import HitlReviewClipRendererV18


_CONFIGURATION_BOOTSTRAP_SUBJECT = "SVC-CONFIGURATION-BOOTSTRAP"
_NATIVE_RESEARCH_BUNDLE = "configs/bundle_ortools_research_native_v3_1.yaml"
_INTELLIGENT_RESEARCH_BUNDLE = (
    "configs/bundle_ortools_research_native_v3_intelligent_validation_5s.yaml"
)


def _close_owned_resources(values: list[Any], *, wait_for_workers: bool) -> None:
    """Best-effort reverse-order cleanup for construction and normal shutdown."""

    seen: set[int] = set()
    for value in values:
        if not isinstance(value, ThreadPoolExecutor) or id(value) in seen:
            continue
        seen.add(id(value))
        try:
            value.shutdown(wait=wait_for_workers, cancel_futures=True)
        except Exception:
            pass
    for value in reversed(values):
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        try:
            close = getattr(value, "close", None)
            if callable(close):
                close()
                continue
            pool_close = getattr(getattr(value, "pool", None), "close", None)
            if callable(pool_close):
                pool_close()
        except Exception:
            pass


def _own(values: list[Any], value: Any) -> Any:
    values.append(value)
    return value


def _bootstrap_packaged_configuration(
    configuration: VersionedConfigurationService,
    *,
    research_errors: dict[str, ConfigurationIntegrityError] | None = None,
) -> dict[str, Any]:
    """Bootstrap under one service identity and always restore caller context."""

    token = CURRENT_DATABASE_IDENTITY.set(
        DatabaseIdentityContext(
            subject_id=_CONFIGURATION_BOOTSTRAP_SUBJECT,
            role="service",
            request_id="configuration-bootstrap",
        )
    )
    try:
        active = configuration.bootstrap_packaged_bundle(
            actor_id=_CONFIGURATION_BOOTSTRAP_SUBJECT
        )
        for path, environment in (
            (_NATIVE_RESEARCH_BUNDLE, "research-native-v3"),
            (_INTELLIGENT_RESEARCH_BUNDLE, "research-native-v3-intelligent"),
        ):
            try:
                configuration.bootstrap_packaged_bundle(
                    actor_id=_CONFIGURATION_BOOTSTRAP_SUBJECT,
                    bundle_path=path,
                    scope="environment",
                    environment=environment,
                )
            except ConfigurationIntegrityError as exc:
                if research_errors is None:
                    raise
                research_errors[environment] = exc
                logging.getLogger(__name__).error("Configuration unavailable for %s: %s", environment, exc)
        return active
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)


@dataclass(slots=True)
class PlatformContainer:
    root: Path
    data_root: Path
    outputs: Path
    runtime: Path
    database_url: str
    jobs: Any
    executor: ThreadPoolExecutor
    registry: PluginRegistry
    game_store: Any
    learning_store: Any
    agent_registry: AgentProfileRegistry
    games: GameManager
    learning: LearningService
    identity: IdentityService
    sakka: SakkaService
    audit: ReplayAuditService
    difficulty: OnlineDifficultyCalibrator
    challenge_miner: ProgressiveChallengeMiner
    explanations: ExplanationService
    reviews: DecisionReviewService
    canonical_history: CanonicalHistoryStore
    adjudication: AdjudicationService
    live: LiveTableService
    live_broker: Any
    authorization_policy: AuthorizationPolicy
    access_tokens: SignedAccessTokenService
    security_state: Any
    oidc_transactions: PkceTransactionStore
    oidc_providers: dict[str, OidcProviderConfig]
    oidc_callback: OidcCallbackService
    passkeys: PasskeyChallengeService
    websocket_tickets: WebSocketTicketService
    identity_profile: dict[str, Any]
    duplicate_profile: dict[str, Any]
    shared_profile: dict[str, Any]
    hitl_store: Any
    hitl: ReviewGovernanceServiceV18
    hitl_review_renderer: HitlReviewClipRendererV18
    hitl_profile: dict[str, Any]
    certification_profile: dict[str, Any]
    benchmark_profile: dict[str, Any]
    operations_profile: dict[str, Any]
    configuration_store: Any
    configuration: VersionedConfigurationService
    configuration_profile: dict[str, Any]
    performance_profile: dict[str, Any]
    performance: PerformanceService
    capability_matrix: dict[str, Any]
    campaign_control: Any
    challenge_locators: ChallengeLocatorServiceV1
    challenge_publication: ChallengePublicationServiceV1
    foundry_catalogs: FoundryCatalogServiceV1
    deep_analysis: FoundryDeepAnalysisServiceV1
    analysis_candidates: FoundryAnalysisCandidateServiceV1
    campaign_proofs: CampaignProofServiceV1
    organic_screening_profile: OrganicScreeningProfileV1
    finalist_replay: FinalistLaffaReplayV1
    finalist_validation_profile: FinalistValidationProfileV1
    finalist_validations: FinalistValidationJobServiceV1
    foundry_profile: dict[str, Any]
    render_batches: RenderBatchServiceV1
    context_searches: ContextSearchJobServiceV1
    sakka_generations: NaturalSakkaGenerationJobServiceV1
    research_store: Any
    research: ResearchFoundryServiceV1
    research_renders: ResearchRenderPreparationServiceV1
    runtime_bundle: ProfileBundle
    configuration_error: ConfigurationIntegrityError | None
    research_configuration_errors: dict[str, ConfigurationIntegrityError]

    def shutdown(self) -> None:
        _close_owned_resources(
            [
                self.executor,
                self.jobs,
                self.game_store,
                self.learning_store,
                getattr(self.identity, "store", None),
                getattr(self.sakka, "store", None),
                getattr(self.adjudication, "store", None),
                getattr(self.live, "store", None),
                self.hitl_store,
                self.configuration_store,
                self.security_state,
                self.oidc_callback,
                self.live_broker,
                self.canonical_history,
                self.campaign_control,
                self.challenge_locators.store,
                self.foundry_catalogs.store,
                self.research_store,
            ],
            wait_for_workers=False,
        )


def _load_or_create_secret(
    runtime: Path, environment_name: str, file_name: str
) -> bytes:
    configured = os.environ.get(environment_name, "").encode("utf-8")
    if configured:
        if len(configured) < 32:
            raise ValueError(f"{environment_name} must contain at least 32 bytes")
        return configured
    path = runtime / file_name
    if path.exists():
        value = path.read_bytes()
        if len(value) >= 32:
            return value
    import secrets

    value = secrets.token_bytes(48)
    path.write_bytes(value)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value


def build_container(
    *,
    root: str | Path | None = None,
    writable_root: str | Path | None = None,
    database_url: str | None = None,
    allow_unavailable_configuration: bool = False,
) -> PlatformContainer:
    owned_resources: list[Any] = []
    try:
        return _build_container(
            root=root,
            writable_root=writable_root,
            database_url=database_url,
            allow_unavailable_configuration=allow_unavailable_configuration,
            owned_resources=owned_resources,
        )
    except BaseException:
        _close_owned_resources(owned_resources, wait_for_workers=True)
        raise


def _build_container(
    *,
    root: str | Path | None,
    writable_root: str | Path | None,
    database_url: str | None,
    allow_unavailable_configuration: bool,
    owned_resources: list[Any],
) -> PlatformContainer:
    resolved_root = platform_root(root)
    resolved_data = data_root(writable_root)
    if writable_root is not None:
        outputs = (resolved_data / "outputs").resolve()
        runtime = (resolved_data / "runtime").resolve()
    else:
        outputs = (
            Path(os.environ.get("BALOOT_OUTPUTS", resolved_data / "outputs"))
            .expanduser()
            .resolve()
        )
        runtime = (
            Path(os.environ.get("BALOOT_RUNTIME", resolved_data / "runtime"))
            .expanduser()
            .resolve()
        )
    outputs.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    dsn = (
        database_url
        if database_url is not None
        else os.environ.get("BALOOT_DATABASE_URL", "")
    ).strip()

    packaged_bundle = build_bundle(resolved_root)
    configuration_profile = packaged_bundle.component("configuration")
    configuration_store = _own(
        owned_resources,
        PostgresConfigurationStore(dsn)
        if dsn
        else SQLiteConfigurationStore(runtime / "configuration.sqlite3"),
    )
    configuration = VersionedConfigurationService(
        configuration_store, root=resolved_root, profile=configuration_profile
    )
    configuration_error = None
    research_configuration_errors: dict[str, ConfigurationIntegrityError] = {}
    try:
        resolved_configuration = _bootstrap_packaged_configuration(
            configuration,
            research_errors=research_configuration_errors if allow_unavailable_configuration else None,
        )
        bundle = profile_bundle_from_resolved(resolved_configuration, root=resolved_root)
        runtime_identity = resolved_configuration["bundle"]
    except ConfigurationIntegrityError as exc:
        if not allow_unavailable_configuration:
            raise
        logging.getLogger(__name__).error("Runtime configuration unavailable: %s", exc)
        configuration_error = exc
        resolved_configuration = None
        # The HTTP app uses these defaults to construct routes while traffic is blocked.
        bundle = packaged_bundle
        runtime_identity = {"revisionId": "", "version": bundle.bundle_profile["profile"]["version"]}
    belief_policy = BeliefPolicy.from_profile(bundle.component("belief"))
    analysis_policy = AnalysisPolicy.from_profile(
        bundle.component("analysis"),
        belief_policy=belief_policy,
    )
    configuration_profile = bundle.component("configuration")
    configuration = VersionedConfigurationService(
        configuration_store,
        root=resolved_root,
        profile=configuration_profile,
    )

    jobs = _own(
        owned_resources,
        PostgresJobStore(dsn) if dsn else JobStore(runtime / "jobs.sqlite3"),
    )
    runtime_workers = dict(bundle.component("runtime")["workers"])
    configured_background_workers = int(runtime_workers["analysis_concurrency"]) + int(
        runtime_workers["video_concurrency"]
    )
    executor = _own(
        owned_resources,
        ThreadPoolExecutor(
            max_workers=int(
                os.environ.get(
                    "BALOOT_VIDEO_WORKERS",
                    str(configured_background_workers),
                )
            ),
            thread_name_prefix="baloot-video",
        ),
    )
    plugin_policy = PluginPolicy.from_profile(
        bundle.component("runtime").get("plugin_host")
    )
    registry = PluginRegistry(policy=plugin_policy)
    video_renderer = RemotionProfessionalRenderer(
        bundle.video,
        repository_root=resolved_root,
        timeout_seconds=plugin_policy.maximum_timeout_seconds,
    )
    registry.register(video_renderer)
    registry.register(
        MultiAngleComposerV1(
            bundle.video,
            repository_root=resolved_root,
        )
    )
    render_batch_executor = RenderBatchExecutorV1(
        video_renderer,
        maximum_concurrency=int(
            bundle.component("runtime")["workers"]["video_concurrency"]
        ),
        maximum_render_slots=int(
            bundle.component("runtime")["workers"]["video_render_slots"]
        ),
    )
    registry.register(render_batch_executor)
    if os.environ.get("BALOOT_DISCOVER_PLUGINS", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        registry.discover()

    game_store = _own(
        owned_resources,
        PostgresGameStore(dsn) if dsn else SQLiteGameStore(runtime / "games.sqlite3"),
    )
    learning_store = _own(
        owned_resources,
        PostgresLearningStore(dsn)
        if dsn
        else LearningStore(runtime / "learning.sqlite3"),
    )
    agent_registry = AgentProfileRegistry.from_mapping(bundle.component("agents"))
    hidden_cipher = AesGcmPayloadCipher.from_environment()
    canonical_history = _own(
        owned_resources,
        CanonicalHistoryStore(
            runtime / "canonical-history.sqlite3", cipher=hidden_cipher
        ),
    )
    games = GameManager(
        resolved_root,
        game_store,
        learning_store=learning_store,
        agent_registry=agent_registry,
        canonical_history=canonical_history,
        bundle_provider=lambda mode: (
            bundle if mode == "full" else build_bundle(resolved_root, mode=mode)
        ),
    )
    learning = LearningService(
        resolved_root,
        resolved_data / "learning",
        store=learning_store,
        game_manager=games,
    )
    render_batches = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=executor,
        batch_executor=render_batch_executor,
        artifact_store=learning.artifact_store,
        output_root=outputs / "render-batches",
        maximum_attempts=int(bundle.component("database")["jobs"]["maximum_attempts"]),
    )
    games.learning_service = learning

    identity_profile = bundle.component("identity")
    identity_store = _own(
        owned_resources,
        PostgresIdentityStore(dsn)
        if dsn
        else SQLiteIdentityStore(runtime / "identity.sqlite3"),
    )
    identity = IdentityService(
        identity_store,
        session_hours=int(
            identity_profile.get("local_accounts", {}).get("session_hours", 720)
        ),
        guest_hours=int(identity_profile.get("guest", {}).get("session_hours", 168)),
        password_minimum=int(
            identity_profile.get("local_accounts", {}).get("password_minimum", 8)
        ),
    )
    duplicate_profile = bundle.component("duplicate")
    sakka_store = _own(
        owned_resources,
        PostgresSakkaStore(dsn) if dsn else SQLiteSakkaStore(runtime / "sakka.sqlite3"),
    )
    board_signer = HmacBoardSigner.load_or_create(runtime)
    sakka = SakkaService(
        resolved_root,
        sakka_store,
        game_manager=games,
        board_signer=board_signer,
        default_maximum_rounds=int(
            duplicate_profile.get("match", {}).get("maximum_rounds", 16)
        ),
        default_room_players=int(
            duplicate_profile.get("rooms", {}).get("default_max_players", 16)
        ),
        player_names=bundle.component("video")["identity"]["default_player_names"],
    )
    audit = ReplayAuditService(
        resolved_root, store=learning_store, learning_service=learning
    )
    sakka.audit_service = audit
    difficulty = OnlineDifficultyCalibrator(
        learning_store, bundle.component("difficulty")
    )
    explanations = ExplanationService(learning_store, audit, bundle.component("audit"))
    reviews = DecisionReviewService(learning_store, audit)

    hitl_profile = bundle.component("hitl")
    hitl_store = _own(
        owned_resources,
        PostgresReviewGovernanceStore(dsn)
        if dsn
        else SQLiteReviewGovernanceStore(runtime / "hitl.sqlite3"),
    )
    hitl = ReviewGovernanceServiceV18(
        hitl_store,
        learning_store,
        audit,
        hitl_profile,
        cluster_secret=_load_or_create_secret(
            runtime, "BALOOT_HITL_CLUSTER_SECRET", "hitl-cluster.secret"
        ),
    )
    hitl_review_renderer = HitlReviewClipRendererV18(hitl_profile)
    learning.hitl_store = hitl_store

    adjudication_profile = AdjudicationProfile.from_yaml(
        resolved_root / "configs/adjudication/saudi_researched_v1.yaml"
    )
    adjudication_store = _own(
        owned_resources,
        PostgresAdjudicationStore(dsn)
        if dsn
        else SQLiteAdjudicationStore(runtime / "adjudication.sqlite3"),
    )
    adjudication = AdjudicationService(adjudication_store, adjudication_profile)

    live_store = _own(
        owned_resources,
        PostgresLiveStore(dsn) if dsn else SQLiteLiveStore(runtime / "live.sqlite3"),
    )
    redis_url = os.environ.get("BALOOT_REDIS_URL", "").strip()
    live_broker = _own(
        owned_resources,
        RedisLiveEventBroker(redis_url) if redis_url else InMemoryLiveEventBroker(),
    )
    live = LiveTableService(
        live_store, games, broker=live_broker, adjudication=adjudication
    )

    authorization_policy = AuthorizationPolicy()
    access_tokens = SignedAccessTokenService(
        _load_or_create_secret(
            runtime, "BALOOT_ACCESS_TOKEN_SECRET", "access-token.secret"
        ),
        issuer=os.environ.get("BALOOT_TOKEN_ISSUER", "baloot-platform"),
        audience=os.environ.get("BALOOT_TOKEN_AUDIENCE", "baloot-api"),
    )
    security_state = _own(
        owned_resources,
        PostgresSecurityStateStore(dsn)
        if dsn
        else SQLiteSecurityStateStore(runtime / "security-v20.sqlite3"),
    )
    oidc_transactions = PkceTransactionStore(
        ttl_seconds=int(os.environ.get("BALOOT_OIDC_TRANSACTION_TTL", "600")),
        backend=security_state,
    )
    oidc_providers: dict[str, OidcProviderConfig] = {}
    if (
        os.environ.get("BALOOT_OIDC_ISSUER")
        and os.environ.get("BALOOT_OIDC_AUTHORIZATION_ENDPOINT")
        and os.environ.get("BALOOT_OIDC_TOKEN_ENDPOINT")
        and os.environ.get("BALOOT_OIDC_CLIENT_ID")
        and os.environ.get("BALOOT_OIDC_REDIRECT_URI")
    ):
        provider_id = os.environ.get("BALOOT_OIDC_PROVIDER_ID", "default")
        oidc_providers[provider_id] = OidcProviderConfig(
            provider_id=provider_id,
            issuer=os.environ["BALOOT_OIDC_ISSUER"],
            authorization_endpoint=os.environ["BALOOT_OIDC_AUTHORIZATION_ENDPOINT"],
            token_endpoint=os.environ["BALOOT_OIDC_TOKEN_ENDPOINT"],
            client_id=os.environ["BALOOT_OIDC_CLIENT_ID"],
            redirect_uri=os.environ["BALOOT_OIDC_REDIRECT_URI"],
            jwks_uri=os.environ.get("BALOOT_OIDC_JWKS_URI") or None,
            client_secret=os.environ.get("BALOOT_OIDC_CLIENT_SECRET") or None,
        )
    oidc_callback = _own(
        owned_resources, OidcCallbackService(oidc_providers, oidc_transactions)
    )
    passkeys = PasskeyChallengeService(
        rp_id=os.environ.get("BALOOT_WEBAUTHN_RP_ID", "localhost"),
        origin=os.environ.get("BALOOT_WEBAUTHN_ORIGIN", "http://localhost:8080"),
        backend=security_state,
    )
    websocket_tickets = WebSocketTicketService(
        _load_or_create_secret(
            runtime, "BALOOT_WEBSOCKET_TICKET_SECRET", "websocket-ticket.secret"
        ),
        security_state,
        lifetime_seconds=int(os.environ.get("BALOOT_WEBSOCKET_TICKET_TTL", "45")),
    )
    challenge_miner = ProgressiveChallengeMiner(
        resolved_root,
        learning_store,
        learning=learning,
        agent_registry=agent_registry,
        bundle=bundle,
    )
    registry.register(GameRepositoryAdapter(game_store))
    registry.register(HistoryStoreAdapter(learning_store))
    registry.register(LearningBackendAdapter(learning))
    registry.register(challenge_miner)
    tactical_detector = CausalTacticalDetectorV1(
        TacticalDetectorProfileV1.build(
            bundle.component("challenge")["tactical_detectors"]
        )
    )
    registry.register(tactical_detector)
    registry.register(DeterministicBatchExecutorV1())
    performance_profile = bundle.component("performance")
    search_config = dict(performance_profile.get("search") or {})
    preferred_search = str(search_config.get("preferred_backend") or "").strip()
    fallback_search = str(search_config.get("fallback_backend") or "").strip()
    require_preferred_search = bool(search_config.get("require_preferred", False))
    if fallback_search != PythonReferenceSearchBackend.descriptor.id:
        raise ValueError("unsupported exact-search fallback backend")
    exact_search_backend = PythonReferenceSearchBackend()
    if preferred_search == NativeExactSearchBackendV3.descriptor.id:
        try:
            exact_search_backend = NativeExactSearchBackendV3(
                NativeAccelerator(bundle.rules)
            )
        except (NativeUnavailable, OSError):
            if require_preferred_search:
                raise
    elif preferred_search != PythonReferenceSearchBackend.descriptor.id:
        raise ValueError("unsupported preferred exact-search backend")
    registry.register(exact_search_backend)
    challenge_profile = bundle.component("challenge")
    structural_domain = bundle.rules.structural_domain
    play_domain = bundle.rules.play_domain
    organic_screening_profile = OrganicScreeningProfileV1.build(
        challenge_profile["campaign_screening"]
    )
    foundry_simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        structural_domain=structural_domain,
        play_domain=play_domain,
        agent_profile=agent_registry.get(organic_screening_profile.agent_profile_id),
        maximum_deal_attempts=int(
            bundle.component("shared")["schedule"]["deal_attempts_per_maximum_round"]
        ),
        all_pass_dealer_policy=str(
            bundle.component("game")["all_pass"]["dealer_policy"]
        ),
        exact_search_backend=exact_search_backend,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        match_value_profile=bundle.component("match_value"),
    )
    registry.register(foundry_simulator)
    confirmation = challenge_profile["confirmation"]
    stability = challenge_profile["stability"]
    selection_profile = SelectionProfileV1.build(challenge_profile["selection_policy"])
    trace_analysis_profile = TraceAnalysisProfileV1.build(
        bid_samples=int(confirmation["bid_samples"]),
        play_samples=int(confirmation["deep_samples"]),
        exact_allocation_limit=int(confirmation["exact_allocation_limit"]),
        exact_search_max_remaining=int(confirmation["exact_search_remaining"]),
        maximum_sampling_attempts=int(confirmation["maximum_sampling_attempts"]),
        exact_search_max_nodes=int(confirmation["exact_search_max_nodes"]),
        exact_search_max_seconds=float(confirmation["exact_search_max_seconds"]),
        exact_search_max_depth=int(confirmation["exact_search_max_depth"]),
        exact_search_max_cache_entries=int(
            confirmation["exact_search_max_cache_entries"]
        ),
        search_backend_id=exact_search_backend.descriptor.id,
        objective=str(confirmation["objective"]),
        use_conventions=False,
        belief_policy=belief_policy,
    )
    finalist_reanalysis_profile = build_finalist_reanalysis_trace_profile_v1(
        stability,
        confirmation,
        search_backend_id=exact_search_backend.descriptor.id,
        belief_policy=belief_policy,
    )
    trace_identity_inputs = {
        "analysis_policy": analysis_policy,
        "belief_policy": belief_policy,
        "match_value_profile_hash": bundle.components["match_value"].sha256,
        "rules_hash": bundle.rules.sha256,
        "convention_hash": bundle.components["conventions"].sha256,
        "challenge_hash": bundle.components["challenge"].sha256,
        "performance_hash": bundle.components["performance"].sha256,
        "search_backend_descriptor": exact_search_backend.descriptor.to_dict(),
    }
    trace_analysis_identity_hashes = {
        "deep": trace_analysis_composite_identity_hash_v1(
            mode="deep",
            descriptor=StagedTraceAnalyzerV1.descriptor.to_dict(),
            profile_hash=trace_analysis_profile.sha256,
            **trace_identity_inputs,
        ),
        "finalist_reanalysis": trace_analysis_composite_identity_hash_v1(
            mode="finalist_reanalysis",
            descriptor=StagedTraceAnalyzerV1.descriptor.to_dict(),
            profile_hash=finalist_reanalysis_profile.sha256,
            **trace_identity_inputs,
        ),
    }
    candidate_resimulation_profile = CandidateResimulationProfileV1.build(
        challenge_profile["candidate_resimulation"]
    )
    candidate_resimulation_simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        structural_domain=structural_domain,
        play_domain=play_domain,
        agent_profile=agent_registry.get(
            candidate_resimulation_profile.agent_profile_id
        ),
        generation_tier="balanced_discovery",
        maximum_deal_attempts=int(
            bundle.component("shared")["schedule"]["deal_attempts_per_maximum_round"]
        ),
        all_pass_dealer_policy=str(
            bundle.component("game")["all_pass"]["dealer_policy"]
        ),
        exact_search_backend=exact_search_backend,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        match_value_profile=bundle.component("match_value"),
    )
    candidate_resimulation = CandidateResimulationV1(
        candidate_resimulation_simulator,
        candidate_resimulation_profile,
    )
    registry.register(candidate_resimulation)
    finalist_validation_profile = FinalistValidationProfileV1.build(
        challenge_profile["finalist_validation"]
    )
    finalist_agent_profile = agent_registry.get(
        finalist_validation_profile.agent_profile_id
    )
    validate_finalist_analysis_compatibility(
        finalist_agent_profile,
        confirmation,
    )
    finalist_lineup = {
        seat: finalist_agent_profile.id for seat in ("north", "west", "south", "east")
    }
    finalist_lineup_snapshot = agent_registry.snapshot_lineup(finalist_lineup)
    qualification_bundle_path = resolved_root / "release" / "agent-qualification.json"
    if qualification_bundle_path.is_file():
        finalist_agent_qualification = FoundryAgentQualificationBundleV1.load(
            qualification_bundle_path,
            profile=finalist_agent_profile,
            lineup_snapshot=finalist_lineup_snapshot,
            benchmark_profile=bundle.component("benchmark"),
        ).release
    else:
        finalist_agent_qualification = FoundryAgentReleaseV1.build(
            release_id=f"{finalist_agent_profile.id}-configured-candidate",
            profile=finalist_agent_profile,
            lineup_snapshot=finalist_lineup_snapshot,
            benchmark_profile=bundle.component("benchmark"),
        )
    finalist_agent_pilot = FinalistAgentPilotEvidenceV1.load(
        root=resolved_root,
        profile=finalist_agent_profile,
        benchmark_profile=bundle.component("benchmark"),
        evidence_path=resolved_root
        / ".local-state"
        / "evidence"
        / "finalist-preanalysis.json",
        source_manifest_path=(
            resolved_root
            / ".local-state"
            / "evidence"
            / "finalist-source-manifest.json"
        ),
    )
    finalist_agent_agreement = FinalistAgentAgreementV1(
        finalist_agent_profile,
        stability,
        maximum_decisions_per_trace=(
            finalist_validation_profile.maximum_confirmation_decisions_per_trace
        ),
        minimum_confidence=selection_profile.minimum_surrounding_confidence,
        convention=bundle.convention,
        exact_search_backend=exact_search_backend,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        match_value_profile=bundle.component("match_value"),
        primary_trace_analysis_profile=trace_analysis_profile,
        primary_analysis_identity_hash=trace_analysis_identity_hashes["deep"],
        reanalysis_trace_analysis_profile=finalist_reanalysis_profile,
        reanalysis_analysis_identity_hash=(
            trace_analysis_identity_hashes["finalist_reanalysis"]
        ),
    )
    registry.register(finalist_agent_agreement)
    finalist_simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        structural_domain=structural_domain,
        play_domain=play_domain,
        agent_profile=finalist_agent_profile,
        agent_release=finalist_agent_qualification,
        generation_tier="throughput_reference",
        maximum_deal_attempts=int(
            bundle.component("shared")["schedule"]["deal_attempts_per_maximum_round"]
        ),
        all_pass_dealer_policy=str(
            bundle.component("game")["all_pass"]["dealer_policy"]
        ),
        exact_search_backend=exact_search_backend,
        decision_agreement=finalist_agent_agreement,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        match_value_profile=bundle.component("match_value"),
        trace_analysis_profile=trace_analysis_profile,
        analysis_identity_hash=trace_analysis_identity_hashes["deep"],
    )
    finalist_replay = FinalistLaffaReplayV1(
        finalist_simulator,
        require_qualified_agent_release_for_publication=(
            finalist_validation_profile.require_qualified_agent_release_for_publication
        ),
        allow_local_preview=finalist_validation_profile.allow_local_preview,
    )
    registry.register(finalist_replay)
    sakka_generation_profile = NaturalSakkaGenerationProfileV1.build(
        challenge_profile["sakka_generation"]
    )
    natural_sakka_generator = NaturalSakkaGeneratorV1(
        foundry_simulator, sakka_generation_profile
    )
    registry.register(natural_sakka_generator)
    trace_import_factory = TraceImportCatalogFactoryV1(
        learning.artifact_store,
        bundle.rules,
        bundle_hash=bundle.sha256,
        engine_version=foundry_simulator.engine_version,
        policy=TraceImportPolicyV1.build(
            bundle.component("challenge")["trace_import_policy"]
        ),
    )
    registry.register(trace_import_factory)
    score_reachability_factory = ScoreReachabilityCatalogFactoryV1(
        bundle.rules,
        ScoreReachabilityProfileV1.build(
            bundle.component("challenge")["score_reachability"]
        ),
    )
    registry.register(score_reachability_factory)
    deep_analysis_profile = DeepAnalysisProfileV1.build(
        challenge_profile["deep_analysis"]
    )
    decision_executor_config = dict(challenge_profile["decision_analysis_executor"])

    def decision_executor_profile(name: str) -> DecisionAnalysisExecutorProfileV1:
        value = dict(decision_executor_config[name])
        return DecisionAnalysisExecutorProfileV1.build(
            worker_count=int(value["workerCount"]),
            maximum_pending_per_worker=int(value["maximumPendingPerWorker"]),
            maximum_seconds=float(value["maximumSeconds"]),
            poll_seconds=float(value["pollSeconds"]),
        )

    screening_executor_profile = decision_executor_profile("screening")
    confirmation_executor_profile = decision_executor_profile("confirmation")
    finalist_executor_profile = decision_executor_profile("finalistReanalysis")
    deep_trace_analyzer = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        trace_analysis_profile,
        match_value_profile=bundle.component("match_value"),
        critical_threshold=float(
            bundle.component("challenge")["publication"]["minimum_publishability"]
        )
        * 100.0,
        minimum_critical_confidence=float(
            confirmation["minimum_confirmation_confidence"]
        ),
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        exact_search_backend=exact_search_backend,
        executor_profile=confirmation_executor_profile,
        cohort_authority_hash=trace_analysis_identity_hashes["deep"],
    )
    registry.register(deep_trace_analyzer)
    finalist_reanalysis_analyzer = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        finalist_reanalysis_profile,
        match_value_profile=bundle.component("match_value"),
        critical_threshold=float(
            bundle.component("challenge")["publication"]["minimum_publishability"]
        )
        * 100.0,
        minimum_critical_confidence=float(
            confirmation["minimum_confirmation_confidence"]
        ),
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        exact_search_backend=exact_search_backend,
        executor_profile=finalist_executor_profile,
        cohort_authority_hash=(trace_analysis_identity_hashes["finalist_reanalysis"]),
    )
    screening = bundle.component("challenge")["screening"]
    screening_trace_analysis_profile = TraceAnalysisProfileV1.build(
        bid_samples=int(screening["shallow_samples"]),
        play_samples=int(screening["shallow_samples"]),
        exact_allocation_limit=int(screening["shallow_exact_allocation_limit"]),
        exact_search_max_remaining=int(screening["shallow_exact_search_remaining"]),
        maximum_sampling_attempts=int(screening["maximum_sampling_attempts"]),
        exact_search_max_nodes=int(screening["exact_search_max_nodes"]),
        exact_search_max_seconds=float(screening["exact_search_max_seconds"]),
        exact_search_max_depth=int(confirmation["exact_search_max_depth"]),
        exact_search_max_cache_entries=int(
            confirmation["exact_search_max_cache_entries"]
        ),
        search_backend_id=exact_search_backend.descriptor.id,
        objective=str(confirmation["objective"]),
        use_conventions=False,
    )
    shallow_trace_analyzer = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        screening_trace_analysis_profile,
        match_value_profile=bundle.component("match_value"),
        critical_threshold=float(
            bundle.component("challenge")["publication"]["minimum_publishability"]
        )
        * 100.0,
        minimum_critical_confidence=0.0,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        exact_search_backend=exact_search_backend,
        executor_profile=screening_executor_profile,
    )
    cascaded_trace_analysis_profile = CascadedTraceAnalysisProfileV1.build(
        screening_trace_analysis_profile,
        trace_analysis_profile,
        maximum_confirmation_decisions_per_trace=(
            deep_analysis_profile.maximum_confirmation_decisions_per_trace
        ),
        minimum_screening_score=deep_analysis_profile.minimum_screening_score,
    )
    cascaded_trace_analyzer = CascadedTraceAnalyzerV1(
        shallow_trace_analyzer,
        deep_trace_analyzer,
        cascaded_trace_analysis_profile,
    )
    registry.register(cascaded_trace_analyzer)
    finalist_cascaded_trace_analysis_profile = CascadedTraceAnalysisProfileV1.build(
        screening_trace_analysis_profile,
        trace_analysis_profile,
        maximum_confirmation_decisions_per_trace=(
            finalist_validation_profile.maximum_confirmation_decisions_per_trace
        ),
        minimum_screening_score=deep_analysis_profile.minimum_screening_score,
    )
    finalist_cascaded_trace_analyzer = CascadedTraceAnalyzerV1(
        shallow_trace_analyzer,
        deep_trace_analyzer,
        finalist_cascaded_trace_analysis_profile,
    )
    disagreement_search = DisagreementSearchV1(
        foundry_simulator,
        shallow_trace_analyzer,
        deep_trace_analyzer,
        DisagreementSearchProfileV1.build(
            bundle.component("challenge")["disagreement_search"]
        ),
    )
    registry.register(disagreement_search)
    registry.register(NaturalHistoryContextVerifierV1())
    critical_context_search = CriticalContextSearchV1(
        foundry_simulator,
        deep_trace_analyzer,
        CriticalContextSearchProfileV1.build(
            bundle.component("challenge")["context_search"]
        ),
    )
    registry.register(critical_context_search)
    diversity_selector = QualityDiversitySelectorV1()
    registry.register(diversity_selector)
    actual_deal_oracle = ActualDealOracleVerifierV1(
        bundle.rules,
        maximum_remaining_cards=(deep_analysis_profile.maximum_oracle_remaining_cards),
        maximum_nodes=int(confirmation["exact_search_max_nodes"]),
        maximum_seconds=float(confirmation["exact_search_max_seconds"]),
        maximum_depth=int(confirmation["exact_search_max_depth"]),
        maximum_cache_entries=int(confirmation["exact_search_max_cache_entries"]),
        detector=tactical_detector,
    )
    registry.register(actual_deal_oracle)
    guided_search_profile = GuidedSearchProfileV1.build(
        bundle.component("challenge")["guided_search"]
    )
    constraint_deal_generator = OrToolsCpSatDealGeneratorV1()
    registry.register(constraint_deal_generator)
    auction_compiler = OrToolsCpSatAuctionCompilerV1()
    registry.register(auction_compiler)
    escalation_compiler = OrToolsCpSatEscalationCompilerV1()
    registry.register(escalation_compiler)
    project_compiler = OrToolsCpSatProjectCompilerV1()
    registry.register(project_compiler)
    play_compiler = OrToolsCpSatPlayCompilerV1()
    registry.register(play_compiler)
    if guided_search_profile.agent_profile_id == foundry_simulator.agent_profile.id:
        guided_search_simulator = foundry_simulator
    elif (
        guided_search_profile.agent_profile_id
        == candidate_resimulation_simulator.agent_profile.id
    ):
        guided_search_simulator = candidate_resimulation_simulator
    elif guided_search_profile.agent_profile_id == finalist_simulator.agent_profile.id:
        guided_search_simulator = finalist_simulator
    else:
        raise ValueError(
            "guided search agent profile must match a governed simulation tier"
        )
    guided_search = GuidedLegalSearchV1(
        guided_search_simulator,
        guided_search_profile,
        opportunity_scanner=actual_deal_oracle,
        deal_generator=constraint_deal_generator,
    )
    registry.register(guided_search)
    thrill_profile = ThrillProfileV1.build(challenge_profile["thrill_profile"])
    challenge_scorer = ProfileChallengeScorerV1()
    registry.register(challenge_scorer)
    perspective_profile = PerspectivePolicyV1.build(
        challenge_profile["perspective_policy"]
    )
    registry.register(
        MultiPerspectiveFoundryV1(
            learning_store,
            critical_threshold=float(
                challenge_miner.profile["publication"]["minimum_publishability"]
            )
            * 100.0,
            perspective_policy=perspective_profile,
        )
    )
    diversity_config = challenge_profile["diversity"]
    diversity_profile = QualityDiversityProfileV1.build(
        {
            "schemaVersion": 1,
            "cellDimensions": diversity_config["cell_dimensions"],
            "topKPerCell": diversity_config["top_k_per_cell"],
            "rejectedAuditModulus": diversity_config["rejected_audit_modulus"],
        }
    )
    analysis_funnel = FoundryAnalysisFunnelV1(
        learning.artifact_store,
        cascaded_trace_analyzer,
        actual_deal_oracle,
        selection_profile,
        diversity_profile,
        scorer=challenge_scorer,
        selector=diversity_selector,
        maximum_oracle_decisions_per_trace=(
            deep_analysis_profile.maximum_oracle_decisions_per_trace
        ),
        allow_pending_surrounding_nominations=(
            deep_analysis_profile.allow_pending_surrounding_nominations
        ),
    )
    registry.register(analysis_funnel)
    finalist_analysis_funnel = FoundryAnalysisFunnelV1(
        learning.artifact_store,
        AdaptiveFinalistTraceAnalyzerV1(
            finalist_cascaded_trace_analyzer,
            finalist_reanalysis_analyzer,
            selection_profile,
            FinalistSurroundingAdjudicationProfileV1.build(
                stability,
                primary_profile_hash=finalist_cascaded_trace_analyzer.profile.sha256,
                reanalysis_profile_hash=finalist_reanalysis_profile.sha256,
                maximum_decisions_per_trace=(
                    finalist_validation_profile.maximum_confirmation_decisions_per_trace
                ),
                minimum_confidence=selection_profile.minimum_surrounding_confidence,
                minimum_quality=selection_profile.minimum_surrounding_quality,
            ),
        ),
        actual_deal_oracle,
        selection_profile,
        diversity_profile,
        scorer=challenge_scorer,
        selector=diversity_selector,
        maximum_oracle_decisions_per_trace=(
            deep_analysis_profile.maximum_oracle_decisions_per_trace
        ),
        allow_pending_surrounding_nominations=False,
    )
    registry.register(audit)
    registry.register(difficulty)
    database_profile = bundle.component("database")
    maximum_campaign_attempts = int(database_profile["jobs"]["maximum_attempts"])
    campaign_control = _own(
        owned_resources,
        PostgresCampaignControlStore(dsn, maximum_attempts=maximum_campaign_attempts)
        if dsn
        else SQLiteCampaignControlStore(
            runtime / "foundry-campaigns.sqlite3",
            maximum_attempts=maximum_campaign_attempts,
        ),
    )
    organic_audit_traces = ResearchAuditTraceRepositoryV1(
        store=campaign_control,
        artifact_store=learning.artifact_store,
    )
    deep_analysis = FoundryDeepAnalysisServiceV1(
        jobs,
        campaign_control,
        analysis_funnel,
        deep_analysis_profile,
        maximum_attempts=maximum_campaign_attempts,
        rules_hash=bundle.rules.sha256,
        bundle_hash=bundle.sha256,
        engine_version=foundry_simulator.engine_version,
        perspective_profile_hash=perspective_profile.sha256,
    )
    locator_key = os.environ.get("BALOOT_CHALLENGE_LOCATOR_KEY", "").encode("utf-8")
    if dsn and len(locator_key) < 32:
        raise ValueError(
            "BALOOT_CHALLENGE_LOCATOR_KEY with at least 32 bytes is required in multi-node mode"
        )
    if not locator_key:
        locator_key = _load_or_create_secret(
            runtime,
            "BALOOT_CHALLENGE_LOCATOR_KEY",
            "challenge-locator.key",
        )
    locator_store = _own(
        owned_resources,
        PostgresChallengeLocatorStore(dsn)
        if dsn
        else SQLiteChallengeLocatorStore(runtime / "challenge-locators.sqlite3"),
    )

    def frozen_research_presentation(result_code: str) -> dict[str, Any]:
        result = research.get_result(str(result_code).upper())
        if not isinstance(result, Mapping):
            raise ValueError("Research challenge result authority is unavailable")
        recipe = research.get_recipe(str(result.get("recipeCode") or ""))
        if not isinstance(recipe, Mapping):
            raise ValueError("Research challenge recipe authority is unavailable")
        resources = dict(recipe.get("resources") or {})
        context = research_execution_contexts.resolve_authority(
            dict(recipe.get("configuration") or {}),
            analysis_mode=str(resources.get("analysisMode") or ""),
        )
        video_artifact = context.bundle.components["video"]
        video = video_artifact.data
        profile = dict(video.get("profile") or {})
        render = dict(video.get("render") or {})
        return {
            "schemaVersion": 1,
            "bundleRevisionId": context.bundle_revision_id,
            "locale": str(render.get("locale") or ""),
            "direction": str(render.get("direction") or ""),
            "profileId": str(profile.get("id") or ""),
            "profileVersion": str(profile.get("version") or ""),
            "profileSha256": video_artifact.sha256,
            "challengeCopy": dict(video.get("challenge_copy") or {}),
        }

    challenge_locators = ChallengeLocatorServiceV1(
        locator_store,
        rules=bundle.rules,
        secret=locator_key,
        public_base_url=os.environ.get(
            "BALOOT_PUBLIC_BASE_URL",
            "http://127.0.0.1:8080",
        ),
        difficulty_calibrator=difficulty,
        friend_ids=lambda subject_id: (
            item["subjectId"] for item in identity.friends(subject_id)
        ),
        challenge_copy=bundle.component("video")["challenge_copy"],
        research_presentation_resolver=frozen_research_presentation,
    )
    challenge_publication = ChallengePublicationServiceV1(
        learning.artifact_store,
        challenge_locators,
        ChallengeAnswerPolicyV1.build(bundle.component("challenge")["answer_policy"]),
        player_names=bundle.component("video")["identity"]["default_player_names"],
    )
    registry.register(challenge_publication)
    analysis_candidates = FoundryAnalysisCandidateServiceV1(
        deep_analysis=deep_analysis,
        campaign_store=campaign_control,
        artifact_store=learning.artifact_store,
        publication_service=challenge_publication,
        renderer=video_renderer,
        rules=bundle.rules,
    )
    registry.register(analysis_candidates)
    finalist_validations = FinalistValidationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        replay=finalist_replay,
        profile=finalist_validation_profile,
        candidates=analysis_candidates,
        funnel=finalist_analysis_funnel,
        artifact_store=learning.artifact_store,
        maximum_attempts=maximum_campaign_attempts,
        thrill_profile=thrill_profile,
        scorer=challenge_scorer,
        candidate_resimulation=candidate_resimulation,
        tactical_viability_probe=(
            actual_deal_oracle
            if finalist_validation_profile.enable_exact_tactical_viability_precheck
            and selection_profile.require_verified_tactical_play
            else None
        ),
        audit_repository=organic_audit_traces,
    )
    foundry_catalog_store = _own(
        owned_resources,
        PostgresFoundryCatalogStoreV1(dsn)
        if dsn
        else SQLiteFoundryCatalogStoreV1(runtime / "foundry-catalogs.sqlite3"),
    )
    if dsn and not os.environ.get("BALOOT_CURATED_SAKKA_SIGNING_KEY", "").strip():
        raise ValueError(
            "BALOOT_CURATED_SAKKA_SIGNING_KEY is required for multi-node storage"
        )
    foundry_catalogs = FoundryCatalogServiceV1(
        foundry_catalog_store,
        learning.artifact_store,
        trace_import_factory,
        score_reachability_factory,
        CuratedSakkaSignerV1.from_secret(
            _load_or_create_secret(
                runtime,
                "BALOOT_CURATED_SAKKA_SIGNING_KEY",
                "curated-sakka-signing.key",
            )
        ),
    )
    context_searches = ContextSearchJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        search_engine=critical_context_search,
        catalogs=foundry_catalogs,
        maximum_attempts=maximum_campaign_attempts,
    )
    sakka_generations = NaturalSakkaGenerationJobServiceV1(
        jobs=jobs,
        background_executor=executor,
        generator=natural_sakka_generator,
        catalogs=foundry_catalogs,
        maximum_attempts=maximum_campaign_attempts,
    )
    campaign_proofs = CampaignProofServiceV1(
        campaign_control,
        foundry_catalogs,
        CampaignProofProfileV1.build(challenge_profile["campaign_proof"]),
    )
    video_production_batch_profile = HookedLaffaBatchProfileV1.build(
        bundle.component("challenge")["video_production_batch"]
    )
    foundry_profile = {
        "schemaVersion": 1,
        "runtimeBundleRevisionId": runtime_identity["revisionId"],
        "runtimeBundleVersion": runtime_identity["version"],
        "rulesHash": bundle.rules.sha256,
        "bundleHash": bundle.sha256,
        "engineVersion": foundry_simulator.engine_version,
        "analysisProfileHash": cascaded_trace_analysis_profile.sha256,
        "searchBackend": exact_search_backend.descriptor.to_dict(),
        "matchValueProfile": {
            "id": str(bundle.component("match_value")["profile"]["id"]),
            "version": str(bundle.component("match_value")["profile"]["version"]),
            "solver": str(bundle.component("match_value")["future_rounds"]["solver"]),
            "sha256": bundle.components["match_value"].sha256,
        },
        "perspectiveProfileHash": perspective_profile.sha256,
        "selectionProfileHash": selection_profile.sha256,
        "agentRelease": foundry_simulator.agent_release,
        "seatAgents": {
            seat: foundry_simulator.agent_release
            for seat in ("north", "west", "south", "east")
        },
        "staticLaneProfileHashes": {
            "organic": organic_screening_profile.sha256,
            "guided": guided_search.profile.sha256,
            "disagreement": disagreement_search.profile.sha256,
        },
        "laneAgentReleases": {
            "organic": foundry_simulator.agent_release,
            "guided": guided_search.simulator.agent_release,
            "disagreement": foundry_simulator.agent_release,
        },
        "laneSeatAgents": {
            "organic": {
                seat: foundry_simulator.agent_release
                for seat in ("north", "west", "south", "east")
            },
            "guided": {
                seat: guided_search.simulator.agent_release
                for seat in ("north", "west", "south", "east")
            },
            "disagreement": {
                seat: foundry_simulator.agent_release
                for seat in ("north", "west", "south", "east")
            },
        },
        "organicScreeningProfile": organic_screening_profile.to_dict(),
        "guidedSearchProfile": guided_search.profile.to_dict(),
        "constraintDealGenerator": {
            "descriptor": constraint_deal_generator.descriptor.to_dict(),
            "available": constraint_deal_generator.available(),
        },
        "candidateResimulationProfile": candidate_resimulation_profile.to_dict(),
        "candidateResimulationAgentRelease": (
            candidate_resimulation_simulator.agent_release
        ),
        "finalistValidationProfile": finalist_validation_profile.to_dict(),
        "finalistAgentRelease": finalist_simulator.agent_release,
        "finalistAgentQualification": finalist_agent_qualification.to_dict(),
        "finalistAgentPilotEvidence": finalist_agent_pilot.to_dict(),
        "finalistSeatAgents": {
            seat: finalist_simulator.agent_release
            for seat in ("north", "west", "south", "east")
        },
        "catalogRequiredLanes": ["historical", "expert"],
        "deepAnalysisProfile": deep_analysis_profile.to_dict(),
        "decisionAnalysisExecutorProfile": {
            "schemaVersion": int(decision_executor_config["schemaVersion"]),
            "id": str(decision_executor_config["id"]),
            "revision": str(decision_executor_config["revision"]),
            "screening": screening_executor_profile.to_dict(),
            "confirmation": confirmation_executor_profile.to_dict(),
            "finalistReanalysis": finalist_executor_profile.to_dict(),
        },
        "campaignProofProfile": campaign_proofs.profile.to_dict(),
        "sakkaGenerationProfile": sakka_generation_profile.to_dict(),
        "videoProductionBatchProfile": video_production_batch_profile.to_dict(),
    }
    research_store = _own(
        owned_resources,
        PostgresResearchStoreV1(dsn)
        if dsn
        else SQLiteResearchStoreV1(runtime / "research.sqlite3"),
    )
    research_domain = BalootDomainSpecV2.build(
        rules=bundle.rules.data,
        conventions=bundle.convention,
        localization=bundle.component("product"),
        bundle_hash=bundle.sha256,
        plugins=registry.list(),
    )
    domain_compiler = BalootDomainCompilerV1()

    def compile_configuration_domain(resolved: Mapping[str, Any]) -> Any:
        profiles = dict(resolved["profiles"])
        return domain_compiler.compile(
            rules=profiles["rules"],
            conventions=profiles["conventions"],
            localization=profiles["product"],
            bundle_hash=str(resolved["effectiveHash"]),
            plugins=registry.list(),
        )

    configuration.set_bundle_compiler(compile_configuration_domain)
    compiled_research_domain = domain_compiler.compile(
        rules=bundle.rules.data,
        conventions=bundle.convention,
        localization=bundle.component("product"),
        bundle_hash=bundle.sha256,
        plugins=registry.list(),
    )
    research_execution_contexts = ResearchExecutionContextFactoryV1(
        configuration_service=configuration,
        artifact_store=learning.artifact_store,
        research_store=research_store,
        root=resolved_root,
        plugin_capabilities=tuple(registry.list()),
        shared_plugin_descriptors=tuple(
            plugin.descriptor.to_dict()
            for plugin in (
                constraint_deal_generator,
                auction_compiler,
                escalation_compiler,
                project_compiler,
                play_compiler,
            )
        ),
        active_bundle=bundle,
        active_exact_search_backend=exact_search_backend,
    )
    if resolved_configuration is not None:
        _, active_research_context = research_execution_contexts.context_from_resolved(
            resolved_configuration
        )
        if (
            active_research_context.compiled_domain.compiled.sha256
            != compiled_research_domain.compiled.sha256
        ):
            raise ValueError("active Research compiled-domain identity mismatch")
    research = ResearchFoundryServiceV1(
        domain=research_domain,
        compiled_domain=compiled_research_domain,
        store=research_store,
        artifact_store=learning.artifact_store,
        executor=executor,
        deal_generator=constraint_deal_generator,
        auction_compiler=auction_compiler,
        escalation_compiler=escalation_compiler,
        project_compiler=project_compiler,
        play_compiler=play_compiler,
        simulator=guided_search_simulator,
        analyzers={
            "screening": shallow_trace_analyzer,
            "cascaded": cascaded_trace_analyzer,
            "deep": deep_trace_analyzer,
        },
        score_profile=thrill_profile,
        decision_score_profile=selection_profile,
        tactical_verifier=actual_deal_oracle,
        maximum_tactical_decisions=(
            deep_analysis_profile.maximum_oracle_decisions_per_trace
        ),
        comparison_profile=challenge_profile["research_comparison"],
        execution_context_factory=research_execution_contexts,
    )
    research.audit_traces = CompositeResearchAuditTraceRepositoryV1(
        research.audit_traces,
        secondary_repositories=(organic_audit_traces,),
    )
    registry.register(research)
    research_renders = ResearchRenderPreparationServiceV1(
        research=research,
        artifact_store=learning.artifact_store,
        publication_service=challenge_publication,
        renderer=video_renderer,
        rules=bundle.rules,
        learning_store=learning_store,
        locator_service=challenge_locators,
        configuration_store=configuration_store,
    )
    registry.register(research_renders)
    performance = PerformanceService(performance_profile)
    capability_matrix = yaml.safe_load(
        (resolved_root / "configs/capabilities/platform_v20.yaml").read_text(
            encoding="utf-8"
        )
    )

    return PlatformContainer(
        resolved_root,
        resolved_data,
        outputs,
        runtime,
        dsn,
        jobs,
        executor,
        registry,
        game_store,
        learning_store,
        agent_registry,
        games,
        learning,
        identity,
        sakka,
        audit,
        difficulty,
        challenge_miner,
        explanations,
        reviews,
        canonical_history,
        adjudication,
        live,
        live_broker,
        authorization_policy,
        access_tokens,
        security_state,
        oidc_transactions,
        oidc_providers,
        oidc_callback,
        passkeys,
        websocket_tickets,
        identity_profile,
        duplicate_profile,
        bundle.component("shared"),
        hitl_store,
        hitl,
        hitl_review_renderer,
        hitl_profile,
        bundle.component("certification"),
        bundle.component("benchmark"),
        bundle.component("operations"),
        configuration_store,
        configuration,
        configuration_profile,
        performance_profile,
        performance,
        capability_matrix,
        campaign_control,
        challenge_locators,
        challenge_publication,
        foundry_catalogs,
        deep_analysis,
        analysis_candidates,
        campaign_proofs,
        organic_screening_profile,
        finalist_replay,
        finalist_validation_profile,
        finalist_validations,
        foundry_profile,
        render_batches,
        context_searches,
        sakka_generations,
        research_store,
        research,
        research_renders,
        bundle,
        configuration_error,
        research_configuration_errors,
    )
