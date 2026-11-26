"""Credential and access policy shared by the HTTP and gRPC services."""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from baloot_platform.identity import Identity, IdentityService
from baloot_platform.security import AccessClaims, AuthorizationPolicy, Scope, SignedAccessTokenService

DEFAULT_TENANT = "default"

PUBLIC_HTTP_READS = frozenset({
    "/", "/app", "/studio", "/foundry", "/configuration-studio", "/play",
    "/guide", "/learn", "/replay", "/challenges", "/decision-workbench",
    "/hitl-studio", "/docs", "/redoc", "/openapi.json", "/favicon.ico",
    "/healthz", "/readyz", "/api/v1/health", "/api/v1/options",
    "/api/v1/engine-guide", "/api/v1/foundry-copy", "/api/v1/challenge-copy",
    "/api/v1/auth/me", "/api/v1/security/capabilities", "/api/v1/capabilities",
})
PUBLIC_HTTP_BOOTSTRAP = frozenset({
    "/api/v1/auth/guest", "/api/v1/auth/register", "/api/v1/auth/login",
    "/api/v1/security/oidc:begin", "/api/v1/security/oidc:callback",
})
PUBLIC_HTTP_ASSETS = (
    "/app-assets/", "/play-assets/", "/guide-assets/", "/foundry-assets/",
    "/foundry-fonts/", "/card-assets/", "/challenge-assets/",
    "/challenge-card-faces/", "/challenge-video-assets/",
)
PUBLIC_CHALLENGE_READ = re.compile(
    r"/c/[^/]+|/api/v2/challenge-locators/[^/:]+(?:/attempt)?"
)

HISTORY_GAME_REOPEN_PATH = "/api/v1/history/games/{identifier}:reopen"
HISTORY_GAME_AUDIT_PATH = "/api/v1/history/games/{identifier}:audit"
_JOB_HTTP = re.compile(r"/api/v1/(?:jobs/[^/:]+|video-jobs/[^/:]+(?:/events)?)")
_ARCHIVE_HTTP_READ = re.compile(
    r"/api/v1/history/games/[^/:]+(?:/(?:replay|audits|verification))?"
)
_JOB_RPC = "/baloot.v2.BalootPlatform/GetJob"


def public_http(method: str, path: str) -> bool:
    if method == "POST":
        return path in PUBLIC_HTTP_BOOTSTRAP
    return method in {"GET", "HEAD"} and (
        path in PUBLIC_HTTP_READS
        or path.startswith(PUBLIC_HTTP_ASSETS)
        or PUBLIC_CHALLENGE_READ.fullmatch(path) is not None
    )


def http_scope(method: str, path: str) -> Scope:
    if method in {"GET", "HEAD"} and _ARCHIVE_HTTP_READ.fullmatch(path):
        return Scope.READ
    if path.startswith("/api/v1/hitl/") and path.endswith("/review-clip"):
        return Scope.MODEL_ADMIN
    if (
        path == "/metrics"
        or path == "/outputs" or path.startswith("/outputs/")
        or path.startswith((
            "/api/v1/jobs/", "/api/v1/video-jobs", "/api/v1/videos/",
            "/api/v1/challenge-jobs", "/api/v1/observability/", "/api/v1/history/games",
        ))
        or (path.startswith("/api/v2/") and not (
            path.startswith("/api/v2/challenge-locators/")
            and path.endswith("/attempts")
        ))
        or (path.startswith("/api/v1/history/") and path.endswith(
            ("/video-jobs", "/video-records", "/videos")
        ))
        or (path.startswith("/api/v1/challenges/") and path.endswith(":render"))
    ):
        return Scope.CONFIG_ADMIN
    return Scope.READ if method in {"GET", "HEAD", "OPTIONS"} else Scope.PLAY


# None marks an explicitly public method. A missing entry has no policy.
RPC_POLICY: dict[str, dict[str, Scope | None]] = {
    "/baloot.v2.BalootPlatform": {
        "HealthCheck": None,
        "CreateGame": Scope.PLAY,
        "GetGame": Scope.READ,
        "SubmitBid": Scope.PLAY,
        "PlayCard": Scope.PLAY,
        "ValidateContext": Scope.PLAY,
        "ContinueContext": Scope.PLAY,
        "Analyze": Scope.PLAY,
        "CreateVideo": Scope.CONFIG_ADMIN,
        "GetJob": Scope.CONFIG_ADMIN,
        "GetReplay": Scope.READ,
        "ReopenGame": http_scope("POST", HISTORY_GAME_REOPEN_PATH),
        "ForkDecision": Scope.PLAY,
        "AuditGame": http_scope("POST", HISTORY_GAME_AUDIT_PATH),
        "ExplainDecision": Scope.PLAY,
        "CompareDecision": Scope.READ,
        "GetDecisionDifficulty": Scope.READ,
        "RecordDecisionAttempt": Scope.PLAY,
        "GetChallengeDifficulty": Scope.READ,
        "RecordChallengeAttempt": Scope.PLAY,
        "GetAdaptiveChallenge": Scope.READ,
        "GetVideoRecord": Scope.CONFIG_ADMIN,
        "GetVideoTimeline": Scope.CONFIG_ADMIN,
        "GetVideoAuditMap": Scope.CONFIG_ADMIN,
        "RenderGameReplay": Scope.CONFIG_ADMIN,
        "CreateRenderBatch": Scope.CONFIG_ADMIN,
        "GetRenderBatch": Scope.CONFIG_ADMIN,
        "CancelRenderBatch": Scope.CONFIG_ADMIN,
    },
    "/baloot.v8.SakkaPlatform": {
        "ListBoards": Scope.READ,
        "CreateBoard": Scope.PLAY,
        "GetBoard": Scope.READ,
        "VerifyBoard": Scope.READ,
        "CreateRoom": Scope.PLAY,
        "GetRoom": Scope.READ,
        "JoinRoom": Scope.PLAY,
        "StartRun": Scope.PLAY,
        "GetRun": Scope.READ,
        "VerifyRun": Scope.READ,
        "PlayAction": Scope.PLAY,
        "AuditRun": Scope.PLAY,
        "ShareDecision": Scope.PLAY,
        "RecordFeedback": Scope.PLAY,
        "GetLeaderboard": Scope.READ,
        "CompareRuns": Scope.PLAY,
    },
    "/baloot.v9.SharedPlatform": {
        "ListBoards": Scope.READ,
        "CreateBoard": Scope.PLAY,
        "GetBoard": Scope.READ,
        "VerifyBoard": Scope.READ,
        "CreateRoom": Scope.PLAY,
        "GetRoom": Scope.READ,
        "JoinRoom": Scope.PLAY,
        "StartRun": Scope.PLAY,
        "GetRun": Scope.READ,
        "VerifyRun": Scope.READ,
        "PlayAction": Scope.PLAY,
        "GetRunReview": Scope.READ,
        "ExplainDecision": Scope.PLAY,
        "GetDecisionReviews": Scope.READ,
        "SubmitDecisionReview": Scope.PLAY,
        "ShareDecision": Scope.PLAY,
        "GetLeaderboard": Scope.READ,
        "CompareRuns": Scope.PLAY,
    },
    "/baloot.v17.BalootOperations": {
        "HealthCheck": None,
        "CreateLiveRoom": Scope.PLAY,
        "GetLiveRoom": Scope.READ,
        "ClaimLiveSeat": Scope.PLAY,
        "SetLiveReady": Scope.PLAY,
        "StartLiveTable": Scope.PLAY,
        "SubmitLiveAction": Scope.PLAY,
        "VerifyLiveRoom": Scope.READ,
        "OpenAdjudicationCase": Scope.PLAY,
        "GetAdjudicationCase": Scope.READ,
        "ListAdjudicationCases": Scope.READ,
        "RecommendAdjudicationRuling": Scope.PLAY,
        "SecurityCapabilities": None,
    },
    "/baloot.v18.HumanLearning": {
        "HealthCheck": None,
        "Profile": Scope.READ,
        "ListCases": Scope.READ,
        "CreateCase": Scope.PLAY,
        "GetCase": Scope.READ,
        "AssignBlindReview": Scope.PLAY,
        "SubmitReview": Scope.PLAY,
        "ComputeConsensus": Scope.PLAY,
        "SetConsent": Scope.PLAY,
        "GetConsent": Scope.READ,
        "CreateTrainingCandidate": Scope.PLAY,
        "RecordChallengerEvaluation": Scope.MODEL_ADMIN,
        "AcceptTrainingCandidate": Scope.MODEL_ADMIN,
        "CreateImprovementProposal": Scope.PLAY,
        "EvaluateImprovement": Scope.MODEL_ADMIN,
        "ApproveImprovement": Scope.MODEL_ADMIN,
        "FinalizeImprovement": Scope.MODEL_ADMIN,
        "RenderReviewClip": Scope.MODEL_ADMIN,
    },
    "/baloot.v19.ConfigurationControl": {
        "GetCapabilities": Scope.READ,
        "ListProfiles": Scope.READ,
        "CreateProfile": Scope.CONFIG_ADMIN,
        "PublishProfile": Scope.CONFIG_ADMIN,
        "RejectProfile": Scope.CONFIG_ADMIN,
        "RetireProfile": Scope.CONFIG_ADMIN,
        "ResolveProfile": Scope.READ,
        "CompareProfiles": Scope.READ,
        "ListBundles": Scope.READ,
        "CreateBundle": Scope.CONFIG_ADMIN,
        "PublishBundle": Scope.CONFIG_ADMIN,
        "RejectBundle": Scope.CONFIG_ADMIN,
        "RetireBundle": Scope.CONFIG_ADMIN,
        "ResolveBundle": Scope.READ,
        "ResolveEffectiveRuntimeConfigV2": Scope.READ,
        "RollbackBundle": Scope.CONFIG_ADMIN,
        "ListPlugins": Scope.READ,
        "GetPerformanceProfile": Scope.READ,
        "RunPerformanceQuick": Scope.CONFIG_ADMIN,
        "GetCapabilityMatrix": Scope.READ,
        "GetCertificationProfile": Scope.READ,
        "VerifyCertificationCandidate": Scope.CONFIG_ADMIN,
        "GetBenchmarkProfile": Scope.READ,
        "EvaluateIntelligenceBenchmark": Scope.CONFIG_ADMIN,
        "GetOperationsProfile": Scope.READ,
    },
}


def rpc_scope(path: str) -> Scope | None:
    service, _, method = path.rpartition("/")
    return RPC_POLICY[service][method]


def tenant_job_http(method: str, path: str) -> bool:
    return method in {"GET", "HEAD"} and _JOB_HTTP.fullmatch(path) is not None


def tenant_job_rpc(path: str) -> bool:
    return path == _JOB_RPC


def history_without_media(game: dict[str, Any]) -> dict[str, Any]:
    return {
        **game,
        "videoRecords": [],
        "decisions": [
            {key: value for key, value in decision.items() if key != "videoRecords"}
            for decision in game.get("decisions") or []
        ],
    }


@dataclass(frozen=True, slots=True)
class Credentials:
    identity: Identity | None
    session: dict[str, Any] | None
    claims: AccessClaims | None
    api_key_valid: bool
    role: str
    tenant_id: str

    @property
    def authenticated(self) -> bool:
        return self.identity is not None or self.claims is not None or self.api_key_valid

    def has_scope(self, policy: AuthorizationPolicy, scope: Scope) -> bool:
        return scope in policy.effective_scopes(role=self.role, claims=self.claims)

    def denial(self, policy: AuthorizationPolicy, scope: Scope, *, tenant_job: bool) -> str | None:
        if not self.has_scope(policy, scope):
            return f"missing required scope: {scope.value}"
        if self.tenant_id != DEFAULT_TENANT and not tenant_job:
            return "the request is outside the tenant scope"
        return None


def claims_identity(claims: AccessClaims, display_name: str) -> Identity:
    role = claims.role
    return Identity(claims.subject_id, role, display_name[:40], None,
                    role in {"service", "admin", "referee"}, None)


def credentials(
    headers: Mapping[str, str],
    token: str | None,
    identities: IdentityService,
    tokens: SignedAccessTokenService,
) -> Credentials:
    identity, session = identities.resolve(token)
    claims = None
    if identity is None and token and token.count(".") == 2:
        try:
            claims = tokens.verify(token)
        except (ValueError, KeyError, TypeError):
            pass
    key = os.environ.get("BALOOT_API_KEY", "")
    supplied = headers.get("x-api-key", "")
    key_valid = bool(key) and hmac.compare_digest(key.encode(), supplied.encode())
    if claims is not None:
        role, tenant = claims.role, claims.tenant_id
    elif key_valid:
        role, tenant = headers.get("x-baloot-role", "service"), None
    elif identity is not None:
        role, tenant = ("guest" if identity.is_guest else "player"), None
    else:
        role, tenant = "anonymous", None
    return Credentials(identity, session, claims, key_valid, role.strip().lower(),
                       (tenant or "").strip() or DEFAULT_TENANT)
