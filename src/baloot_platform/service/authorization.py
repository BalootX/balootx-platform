"""Credential and access policy shared by the HTTP and gRPC services."""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from baloot_platform.identity import Identity, IdentityService
from baloot_platform.security import AccessClaims, Scope, SignedAccessTokenService

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

PUBLIC_RPC = frozenset({
    "/baloot.v2.BalootPlatform/HealthCheck",
    "/baloot.v17.BalootOperations/HealthCheck",
    "/baloot.v17.BalootOperations/SecurityCapabilities",
    "/baloot.v18.HumanLearning/HealthCheck",
})
_JOB_HTTP = re.compile(r"/api/v1/(?:video-)?jobs/[^/:]+")
_JOB_RPC = "/baloot.v2.BalootPlatform/GetJob"
_PLATFORM_OPERATOR_RPC = frozenset({
    "CreateVideo", "CreateRenderBatch", "GetRenderBatch", "CancelRenderBatch",
    "GetVideoRecord", "GetVideoTimeline", "GetVideoAuditMap", "RenderGameReplay",
    "GetReplay",
})
_CONFIG_OPERATOR_RPC = frozenset({
    "CreateProfile", "PublishProfile", "RejectProfile", "RetireProfile",
    "CreateBundle", "PublishBundle", "RejectBundle", "RetireBundle",
    "RollbackBundle", "RunPerformanceQuick", "VerifyCertificationCandidate",
    "EvaluateIntelligenceBenchmark",
})
_MODEL_OPERATOR_RPC = frozenset({
    "RecordChallengerEvaluation", "AcceptTrainingCandidate", "EvaluateImprovement",
    "ApproveImprovement", "FinalizeImprovement", "RenderReviewClip",
})


def public_http(method: str, path: str) -> bool:
    if method == "POST":
        return path in PUBLIC_HTTP_BOOTSTRAP
    return method in {"GET", "HEAD"} and (
        path in PUBLIC_HTTP_READS
        or path.startswith(PUBLIC_HTTP_ASSETS)
        or PUBLIC_CHALLENGE_READ.fullmatch(path) is not None
    )


def http_scope(method: str, path: str) -> Scope:
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


def rpc_scope(path: str) -> Scope:
    service, _, method = path.rpartition("/")
    if (
        path == _JOB_RPC
        or service == "/baloot.v2.BalootPlatform" and method in _PLATFORM_OPERATOR_RPC
        or service == "/baloot.v19.ConfigurationControl" and method in _CONFIG_OPERATOR_RPC
    ):
        return Scope.CONFIG_ADMIN
    if service == "/baloot.v18.HumanLearning" and method in _MODEL_OPERATOR_RPC:
        return Scope.MODEL_ADMIN
    if method == "Profile" or method.startswith(("Get", "List", "Verify", "Resolve", "Compare")):
        return Scope.READ
    return Scope.PLAY


def tenant_job_http(method: str, path: str) -> bool:
    return method in {"GET", "HEAD"} and _JOB_HTTP.fullmatch(path) is not None


def tenant_job_rpc(path: str) -> bool:
    return path == _JOB_RPC


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
