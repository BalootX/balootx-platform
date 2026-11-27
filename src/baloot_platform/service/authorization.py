"""Credential and access policy shared by the HTTP and gRPC services."""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from starlette._utils import get_route_path
from starlette.routing import Match, Router
from starlette.types import Scope as ASGIScope

from baloot_platform.identity import Identity, IdentityService
from baloot_platform.security import AccessClaims, AuthorizationPolicy, Scope, SignedAccessTokenService

DEFAULT_TENANT = "default"

HISTORY_GAME_REOPEN_PATH = "/api/v1/history/games/{identifier}:reopen"
HISTORY_GAME_AUDIT_PATH = "/api/v1/history/games/{identifier}:audit"
_JOB_HTTP = re.compile(r"/api/v1/(?:jobs/[^/:]+|video-jobs/[^/:]+(?:/events)?)")
_JOB_RPC = "/baloot.v2.BalootPlatform/GetJob"


# None marks explicitly public access. Keys use the registered route template.
HTTP_POLICY: dict[tuple[str, str], Scope | None] = {
    ("GET", "/"): None,
    ("POST", "/api/v1/auth/guest"): None,
    ("POST", "/api/v1/auth/login"): None,
    ("POST", "/api/v1/auth/logout"): None,
    ("GET", "/api/v1/auth/me"): None,
    ("POST", "/api/v1/auth/register"): None,
    ("GET", "/api/v1/capabilities"): None,
    ("GET", "/api/v1/challenge-copy"): None,
    ("GET", "/api/v1/engine-guide"): None,
    ("GET", "/api/v1/foundry-copy"): None,
    ("GET", "/api/v1/health"): None,
    ("GET", "/api/v1/options"): None,
    ("GET", "/api/v1/security/capabilities"): None,
    ("POST", "/api/v1/security/oidc:begin"): None,
    ("POST", "/api/v1/security/oidc:callback"): None,
    ("GET", "/api/v2/challenge-locators/{code}"): None,
    ("GET", "/api/v2/challenge-locators/{code}/attempt"): None,
    ("GET", "/app"): None,
    ("GET", "/app-assets"): None,
    ("HEAD", "/app-assets"): None,
    ("GET", "/c/{code}"): None,
    ("GET", "/card-assets"): None,
    ("HEAD", "/card-assets"): None,
    ("GET", "/challenge-assets"): None,
    ("HEAD", "/challenge-assets"): None,
    ("GET", "/challenge-card-faces"): None,
    ("HEAD", "/challenge-card-faces"): None,
    ("GET", "/challenge-video-assets"): None,
    ("HEAD", "/challenge-video-assets"): None,
    ("GET", "/challenges"): None,
    ("GET", "/configuration-studio"): None,
    ("GET", "/decision-workbench"): None,
    ("GET", "/docs"): None,
    ("GET", "/favicon.ico"): None,
    ("GET", "/foundry"): None,
    ("GET", "/foundry-assets"): None,
    ("HEAD", "/foundry-assets"): None,
    ("GET", "/foundry-fonts"): None,
    ("HEAD", "/foundry-fonts"): None,
    ("GET", "/guide"): None,
    ("GET", "/guide-assets"): None,
    ("HEAD", "/guide-assets"): None,
    ("GET", "/healthz"): None,
    ("GET", "/hitl-studio"): None,
    ("GET", "/learn"): None,
    ("GET", "/openapi.json"): None,
    ("HEAD", "/openapi.json"): None,
    ("GET", "/play"): None,
    ("GET", "/play-assets"): None,
    ("HEAD", "/play-assets"): None,
    ("GET", "/readyz"): None,
    ("GET", "/redoc"): None,
    ("GET", "/replay"): None,
    ("GET", "/shared-assets"): None,
    ("HEAD", "/shared-assets"): None,
    ("GET", "/studio"): None,

    ("GET", "/api/v1/adaptive/challenges/next"): Scope.READ,
    ("GET", "/api/v1/adjudication/cases"): Scope.READ,
    ("GET", "/api/v1/adjudication/cases/{case_id}"): Scope.READ,
    ("GET", "/api/v1/adjudication/cases/{case_id}/directive"): Scope.READ,
    ("GET", "/api/v1/adjudication/cases/{case_id}/recommendation"): Scope.READ,
    ("GET", "/api/v1/adjudication/cases/{case_id}/verification"): Scope.READ,
    ("GET", "/api/v1/agents/profiles"): Scope.READ,
    ("GET", "/api/v1/audits/{audit_id}"): Scope.READ,
    ("GET", "/api/v1/benchmark/profile"): Scope.READ,
    ("GET", "/api/v1/bundles/current"): Scope.READ,
    ("GET", "/api/v1/canonical-history/{identifier}/lineage"): Scope.READ,
    ("GET", "/api/v1/certification/profile"): Scope.READ,
    ("GET", "/api/v1/challenges"): Scope.READ,
    ("GET", "/api/v1/challenges/{challenge_id}"): Scope.READ,
    ("GET", "/api/v1/challenges/{challenge_id}/difficulty"): Scope.READ,
    ("GET", "/api/v1/challenges/{challenge_id}/publication-reviews"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/bundles"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/bundles/{revision_id}"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/bundles/{revision_id}:resolve"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/bundles/{revision_id}:resolve-v2"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/bundles/{revision_id}/events"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profile"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profile-schemas/{family}"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profiles"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profiles:compare"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profiles:resolve"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profiles/{revision_id}"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/profiles/{revision_id}/events"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/resolve"): Scope.READ,
    ("GET", "/api/v1/configuration-registry/resolve-v2"): Scope.READ,
    ("GET", "/api/v1/friends"): Scope.READ,
    ("GET", "/api/v1/friends/requests"): Scope.READ,
    ("GET", "/api/v1/games"): Scope.READ,
    ("GET", "/api/v1/games/{game_id}"): Scope.READ,
    ("GET", "/api/v1/games/{game_id}/audit"): Scope.READ,
    ("GET", "/api/v1/games/{game_id}/decisions"): Scope.READ,
    ("GET", "/api/v1/games/{game_id}/versions"): Scope.READ,
    ("GET", "/api/v1/history/decisions/{decision_id}"): Scope.READ,
    ("GET", "/api/v1/history/decisions/{decision_id}/audits"): Scope.READ,
    ("GET", "/api/v1/history/decisions/{decision_id}/comparison"): Scope.READ,
    ("GET", "/api/v1/history/decisions/{decision_id}/difficulty"): Scope.READ,
    ("GET", "/api/v1/history/games/{identifier}"): Scope.READ,
    ("GET", "/api/v1/history/games/{identifier}/audits"): Scope.READ,
    ("GET", "/api/v1/history/games/{identifier}/replay"): Scope.READ,
    ("GET", "/api/v1/history/games/{identifier}/verification"): Scope.READ,
    ("GET", "/api/v1/hitl/cases"): Scope.READ,
    ("GET", "/api/v1/hitl/cases/{case_id}"): Scope.READ,
    ("GET", "/api/v1/hitl/cases/{case_id}/verification"): Scope.READ,
    ("GET", "/api/v1/hitl/consent"): Scope.READ,
    ("GET", "/api/v1/hitl/improvements"): Scope.READ,
    ("GET", "/api/v1/hitl/improvements/{proposal_id}"): Scope.READ,
    ("GET", "/api/v1/hitl/profile"): Scope.READ,
    ("GET", "/api/v1/hitl/reviewers/{subject_id}"): Scope.READ,
    ("GET", "/api/v1/live/rooms"): Scope.READ,
    ("GET", "/api/v1/live/rooms/{identifier}"): Scope.READ,
    ("GET", "/api/v1/live/rooms/{identifier}/verification"): Scope.READ,
    ("GET", "/api/v1/operations/profile"): Scope.READ,
    ("GET", "/api/v1/performance/profile"): Scope.READ,
    ("GET", "/api/v1/platform/capabilities"): Scope.READ,
    ("GET", "/api/v1/plugins"): Scope.READ,
    ("GET", "/api/v1/plugins-legacy"): Scope.READ,
    ("GET", "/api/v1/sakka/boards"): Scope.READ,
    ("GET", "/api/v1/sakka/boards/{identifier}"): Scope.READ,
    ("GET", "/api/v1/sakka/boards/{identifier}/verification"): Scope.READ,
    ("GET", "/api/v1/sakka/rooms/{identifier}"): Scope.READ,
    ("GET", "/api/v1/sakka/rooms/{identifier}/events"): Scope.READ,
    ("GET", "/api/v1/sakka/rooms/{identifier}/leaderboard"): Scope.READ,
    ("GET", "/api/v1/sakka/runs/{run_id}"): Scope.READ,
    ("GET", "/api/v1/sakka/runs/{run_id}/audit"): Scope.READ,
    ("GET", "/api/v1/sakka/runs/{run_id}/verification"): Scope.READ,
    ("GET", "/api/v1/security/passkeys"): Scope.READ,
    ("GET", "/api/v1/shared/boards"): Scope.READ,
    ("GET", "/api/v1/shared/boards/{identifier}"): Scope.READ,
    ("GET", "/api/v1/shared/boards/{identifier}/verification"): Scope.READ,
    ("GET", "/api/v1/shared/rooms/{identifier}"): Scope.READ,
    ("GET", "/api/v1/shared/rooms/{identifier}/events"): Scope.READ,
    ("GET", "/api/v1/shared/rooms/{identifier}/leaderboard"): Scope.READ,
    ("GET", "/api/v1/shared/runs/{run_id}"): Scope.READ,
    ("GET", "/api/v1/shared/runs/{run_id}/decisions/{decision_id}/explanation"): Scope.READ,
    ("GET", "/api/v1/shared/runs/{run_id}/decisions/{decision_id}/reviews"): Scope.READ,
    ("GET", "/api/v1/shared/runs/{run_id}/review"): Scope.READ,
    ("GET", "/api/v1/shared/runs/{run_id}/verification"): Scope.READ,
    ("GET", "/api/v1/users:search"): Scope.READ,
    ("GET", "/api/v2/friends"): Scope.READ,
    ("GET", "/api/v2/friends/requests"): Scope.READ,
    ("GET", "/api/v2/users:search"): Scope.READ,

    ("POST", "/api/v1/adjudication/cases"): Scope.PLAY,
    ("POST", "/api/v1/adjudication/cases/{case_id}/appeals"): Scope.PLAY,
    ("POST", "/api/v1/adjudication/cases/{case_id}/evidence"): Scope.PLAY,
    ("POST", "/api/v1/analyses"): Scope.PLAY,
    ("POST", "/api/v1/challenges/{challenge_id}:approve"): Scope.PLAY,
    ("POST", "/api/v1/challenges/{challenge_id}:publish"): Scope.PLAY,
    ("POST", "/api/v1/challenges/{challenge_id}:retire"): Scope.PLAY,
    ("POST", "/api/v1/challenges/{challenge_id}:status"): Scope.PLAY,
    ("POST", "/api/v1/challenges/{challenge_id}/attempts"): Scope.PLAY,
    ("POST", "/api/v1/challenges/{challenge_id}/publication-reviews"): Scope.PLAY,
    ("POST", "/api/v1/configuration-registry/profiles/{revision_id}:validate"): Scope.PLAY,
    ("POST", "/api/v1/configuration:validate"): Scope.PLAY,
    ("POST", "/api/v1/contexts:continue"): Scope.PLAY,
    ("POST", "/api/v1/contexts:validate"): Scope.PLAY,
    ("POST", "/api/v1/friends/requests"): Scope.PLAY,
    ("POST", "/api/v1/friends/requests/{request_id}:respond"): Scope.PLAY,
    ("POST", "/api/v1/games"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}:archive"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}/actions"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}/bids"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}/bots:advance"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}/deal:annul"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}/rounds:next"): Scope.PLAY,
    ("POST", "/api/v1/games/{game_id}/versions/{version}:branch"): Scope.PLAY,
    ("POST", "/api/v1/history/decisions/{decision_id}:explain"): Scope.PLAY,
    ("POST", "/api/v1/history/decisions/{decision_id}:fork"): Scope.PLAY,
    ("POST", "/api/v1/history/decisions/{decision_id}:reanalyze"): Scope.PLAY,
    ("POST", "/api/v1/history/decisions/{decision_id}/attempts"): Scope.PLAY,
    ("POST", "/api/v1/history/decisions/{decision_id}/feedback"): Scope.PLAY,
    ("POST", "/api/v1/history/decisions/{decision_id}/intent"): Scope.PLAY,
    ("POST", "/api/v1/hitl/cases"): Scope.PLAY,
    ("POST", "/api/v1/hitl/cases/{case_id}/assignments"): Scope.PLAY,
    ("POST", "/api/v1/hitl/cases/{case_id}/consensus"): Scope.PLAY,
    ("POST", "/api/v1/hitl/cases/{case_id}/submissions"): Scope.PLAY,
    ("POST", "/api/v1/hitl/cases/{case_id}/training-candidates"): Scope.PLAY,
    ("PUT", "/api/v1/hitl/consent"): Scope.PLAY,
    ("POST", "/api/v1/hitl/improvements"): Scope.PLAY,
    ("POST", "/api/v1/hitl/improvements/{proposal_id}:approve"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/actions"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/deadline:enforce"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/disconnect"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/disconnects:enforce"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/pause"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/ready"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/reconnect"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/referee"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/resume"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/seats"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/spectators"): Scope.PLAY,
    ("POST", "/api/v1/live/rooms/{identifier}/start"): Scope.PLAY,
    ("POST", "/api/v1/sakka/boards"): Scope.PLAY,
    ("POST", "/api/v1/sakka/comparisons"): Scope.PLAY,
    ("POST", "/api/v1/sakka/rooms"): Scope.PLAY,
    ("POST", "/api/v1/sakka/rooms/{identifier}:join"): Scope.PLAY,
    ("POST", "/api/v1/sakka/rooms/{identifier}/runs"): Scope.PLAY,
    ("POST", "/api/v1/sakka/runs/{run_id}:audit"): Scope.PLAY,
    ("POST", "/api/v1/sakka/runs/{run_id}/actions"): Scope.PLAY,
    ("POST", "/api/v1/sakka/runs/{run_id}/actions/{sequence}:share"): Scope.PLAY,
    ("POST", "/api/v1/sakka/runs/{run_id}/actions/{sequence}/feedback"): Scope.PLAY,
    ("POST", "/api/v1/security/passkeys:authenticate:finish"): Scope.PLAY,
    ("POST", "/api/v1/security/passkeys:begin"): Scope.PLAY,
    ("POST", "/api/v1/security/passkeys:register:finish"): Scope.PLAY,
    ("POST", "/api/v1/security/tokens"): Scope.PLAY,
    ("POST", "/api/v1/security/websocket-tickets"): Scope.PLAY,
    ("POST", "/api/v1/shared/boards"): Scope.PLAY,
    ("POST", "/api/v1/shared/comparisons"): Scope.PLAY,
    ("POST", "/api/v1/shared/rooms"): Scope.PLAY,
    ("POST", "/api/v1/shared/rooms/{identifier}:join"): Scope.PLAY,
    ("POST", "/api/v1/shared/rooms/{identifier}/runs"): Scope.PLAY,
    ("POST", "/api/v1/shared/runs/{run_id}/actions"): Scope.PLAY,
    ("POST", "/api/v1/shared/runs/{run_id}/actions/{sequence}:share"): Scope.PLAY,
    ("POST", "/api/v1/shared/runs/{run_id}/decisions/{decision_id}/reviews"): Scope.PLAY,
    ("POST", "/api/v2/challenge-locators/{code}/attempts"): Scope.PLAY,

    ("GET", "/api/v1/canonical-history/{identifier}/snapshot"): Scope.HIDDEN_READ,
    ("GET", "/api/v1/canonical-history/{identifier}/verification"): Scope.HIDDEN_READ,

    ("POST", "/api/v1/adjudication/cases/{case_id}/appeals:resolve"): Scope.REFEREE,
    ("POST", "/api/v1/adjudication/cases/{case_id}/rulings"): Scope.REFEREE,
    ("POST", "/api/v1/hitl/cases/{case_id}/adjudications"): Scope.REFEREE,

    ("POST", "/api/v1/benchmark/evaluations"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/canonical-history/{identifier}/branches"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/canonical-history/{identifier}/corrections"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/certification/candidates:verify"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/challenge-jobs"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/challenges/{challenge_id}:recalibrate"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/challenges/{challenge_id}:render"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry:bootstrap"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bootstrap"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bundles"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bundles/{revision_id}:compile"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bundles/{revision_id}:publish"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bundles/{revision_id}:reject"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bundles/{revision_id}:retire"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/bundles/{revision_id}:rollback"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/profiles"): Scope.CONFIG_ADMIN,
    ("PATCH", "/api/v1/configuration-registry/profiles/{revision_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/profiles/{revision_id}:publish"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/profiles/{revision_id}:reject"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/configuration-registry/profiles/{revision_id}:retire"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/history/decisions/{decision_id}/video-jobs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/history/decisions/{decision_id}/videos"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/history/games"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/history/games/{identifier}/video-jobs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/history/games/{identifier}/video-records"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/history/games/{identifier}/videos"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/jobs/{job_id}"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/observability/snapshot"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/performance/quick"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/video-jobs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/video-jobs/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v1/video-jobs/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/video-jobs/{job_id}/events"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/videos/{video_id}"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/videos/{video_id}/audit-map"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v1/videos/{video_id}/timeline"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/analysis-batches"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/analysis-batches"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/analysis-batches/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/analysis-batches/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/analysis-batches/{job_id}:execute"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/analysis-batches/{job_id}/candidates"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/analysis-batches/{job_id}/candidates/{candidate_hash}:prepare"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/campaign-equivalence-proofs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/campaigns"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/campaigns"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/campaigns/{campaign_hash}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/campaigns/{campaign_hash}:cancel"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/campaigns/{campaign_hash}:execute"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/campaigns/{campaign_hash}/attempts"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/campaigns/{campaign_hash}/shards"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/challenge-locators"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/challenge-locators/{code}:revoke"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/comparison-campaigns"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/comparison-campaigns"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/comparison-campaigns/{campaign_code}"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/comparison-campaigns/{campaign_code}/events"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/config-revisions"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/configuration/bundles"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/bundles"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/configuration/bundles/{revision_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/bundles/{revision_id}:compile"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/bundles/{revision_id}:publish"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/bundles/{revision_id}:retire"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/configuration/profile-schemas/{family}"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/configuration/profiles"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/profiles"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/configuration/profiles/{revision_id}"): Scope.CONFIG_ADMIN,
    ("PATCH", "/api/v2/configuration/profiles/{revision_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/profiles/{revision_id}:publish"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/profiles/{revision_id}:retire"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/configuration/profiles/{revision_id}:validate"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/context-searches"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/context-searches/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/context-searches/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/context-searches/{job_id}/events"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/corrections"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/corrections"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/datasets"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/datasets"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/domain-capabilities"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/domain-compilations/{compilation_hash}"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/domain-compilations/{compilation_hash}/coverage"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/finalist-validations"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/finalist-validations"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/finalist-validations/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/finalist-validations/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/finalist-validations/{job_id}:prepare-best"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/foundry-catalogs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/foundry-catalogs/{catalog_hash}"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/foundry-profile"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/laffat/{identifier}:prepare-render"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}/complete-trace"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}/decisions"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}/evidence"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}/perspectives"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}/renders"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/laffat/{identifier}/scenario-graph"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/models"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/models"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/plugins"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/publications"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/publications"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/qualified-laffat"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/render-batches"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/render-batches"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/render-batches/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/render-batches/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/render-batches/{job_id}/events"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-exact-traces:preview-legal-actions"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-recipes"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-recipes"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-recipes:estimate"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-recipes:validate"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-recipes/{code}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-recipes/{code}:preview-v4-conversion"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-recipes/{code}:retire"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{code}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-results/{code}:prepare-render"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-results/{code}:verify-play-compiler"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{code}/decisions"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{code}/decisions/{ordinal}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-results/{code}/decisions/{ordinal}:prepare-render"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{code}/exact-state"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{code}/perspectives"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{code}/renders"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{result_code}/operator-audit"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-results/{result_code}/quality-audit"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-runs"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-runs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-runs/{run_code}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-runs/{run_code}:cancel"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/research-runs/{run_code}:render-selected"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-runs/{run_code}/events"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-runs/{run_code}/sequential-search-receipts"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-runs/{run_code}/stats"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/research-stats"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/rule-primitives"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/sakka-generations"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/sakka-generations/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/sakka-generations/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/sakka-generations/{job_id}/events"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/sakkas"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/sakkas/{sakka_hash}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/sakkas/curated"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/sakkas/natural"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/score-reachability-catalogs"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/score-reachability-catalogs/{catalog_hash}:prove"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/semantic-disagreements"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/shards"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/shards:claim"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/shards/{spec_hash}:commit"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/shards/{spec_hash}:heartbeat"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/trace-import-catalogs"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/video-production-batches"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/video-production-batches"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/video-production-batches/{job_id}"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/video-production-batches/{job_id}:cancel"): Scope.CONFIG_ADMIN,
    ("POST", "/api/v2/video-production-batches/{job_id}:resume"): Scope.CONFIG_ADMIN,
    ("GET", "/api/v2/video-production-batches/{job_id}/events"): Scope.CONFIG_ADMIN,
    ("GET", "/metrics"): Scope.CONFIG_ADMIN,
    ("GET", "/outputs"): Scope.CONFIG_ADMIN,
    ("HEAD", "/outputs"): Scope.CONFIG_ADMIN,
    ("POST", HISTORY_GAME_AUDIT_PATH): Scope.CONFIG_ADMIN,
    ("POST", HISTORY_GAME_REOPEN_PATH): Scope.CONFIG_ADMIN,

    ("POST", "/api/v1/hitl/cases/{case_id}/moderation-flags"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/hitl/cases/{case_id}/review-clip"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/hitl/improvements/{proposal_id}:finalize"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/hitl/improvements/{proposal_id}/evaluations"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/hitl/moderation-flags/{flag_id}:resolve"): Scope.MODEL_ADMIN,
    ("PUT", "/api/v1/hitl/reviewers/{subject_id}"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/hitl/training-candidates/{candidate_id}:accept"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/hitl/training-candidates/{candidate_id}/evaluations"): Scope.MODEL_ADMIN,
    ("GET", "/api/v1/learning/datasets"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/learning/datasets"): Scope.MODEL_ADMIN,
    ("GET", "/api/v1/learning/models"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/learning/models:train"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/learning/models/{model_id}:evaluate"): Scope.MODEL_ADMIN,
    ("POST", "/api/v1/learning/models/{model_id}:promote"): Scope.MODEL_ADMIN,
    ("GET", "/api/v1/learning/queue"): Scope.MODEL_ADMIN,
}


def http_scope(method: str, route_path: str) -> Scope | None:
    return HTTP_POLICY.get((method, route_path), Scope.CONFIG_ADMIN)


def http_routes(router: Router) -> Iterator[Any]:
    for route in router.routes:
        # FastAPI can keep included routers grouped instead of copying their routes.
        effective_routes = getattr(route, "effective_route_contexts", None)
        if effective_routes is None:
            yield route
        else:
            for context in effective_routes():
                yield context.starlette_route or context


def http_request_scope(router: Router, asgi_scope: ASGIScope) -> Scope | None:
    method = asgi_scope["method"]
    path = get_route_path(asgi_scope)
    partial = False
    get_route = None
    for route in http_routes(router):
        match, _ = route.matches(asgi_scope)
        if match is Match.FULL:
            return _matched_http_scope(method, route.path, path)
        if match is Match.PARTIAL:
            partial = True
            if get_route is None and "GET" in route.methods:
                get_route = route.path
    if method == "HEAD" and get_route is not None:
        return _matched_http_scope("GET", get_route, path)

    if not partial and router.redirect_slashes and path != "/" and path.endswith("/"):
        redirect_scope = {**asgi_scope, "path": asgi_scope["path"].rstrip("/")}
        for route in http_routes(router):
            if route.matches(redirect_scope)[0] is Match.FULL:
                return _matched_http_scope(method, route.path, path) or Scope.CONFIG_ADMIN
    return Scope.CONFIG_ADMIN


def _matched_http_scope(method: str, route_path: str, path: str) -> Scope | None:
    scope = http_scope(method, route_path)
    if method == "GET" and (
        (route_path.startswith("/api/v2/challenge-locators/") and ":" in path)
        or (
            route_path.startswith("/api/v1/history/games/")
            and (":" in path or path.endswith("/"))
        )
    ):
        return Scope.CONFIG_ADMIN
    return scope


# These mutations require a service key until they bind the resource or actor.
RPC_SERVICE_KEY_METHODS = frozenset({
    "/baloot.v2.BalootPlatform/CreateGame",
    "/baloot.v2.BalootPlatform/ForkDecision",
    "/baloot.v2.BalootPlatform/PlayCard",
    "/baloot.v2.BalootPlatform/RecordChallengeAttempt",
    "/baloot.v2.BalootPlatform/RecordDecisionAttempt",
    "/baloot.v2.BalootPlatform/SubmitBid",
})


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
