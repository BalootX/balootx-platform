"""Dependency-free MCP adapter for Baloot Platform.

The local stdio transport is implemented directly over newline-delimited JSON-RPC.
A stateless Streamable HTTP transport is also provided through FastAPI.  Both
transports delegate to the authoritative platform services and contain no
Baloot rule implementation.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import sys
import threading
from urllib.parse import urlencode
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search_errors import SearchTimeBudgetExceeded
from baloot_platform.config import build_bundle, data_root, platform_root
from baloot_platform.core.enums import Seat
from baloot_platform.core.serialization import continue_trusted_context, replay_trusted_context
from baloot_platform.version import __version__
from baloot_platform.security import Scope, SignedAccessTokenService
from baloot_platform.security.oidc import load_oidc_providers

ROOT = platform_root()
BELIEF_POLICY = BeliefPolicy.from_profile(root=ROOT)
SERVER_VERSION = __version__
LATEST_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = {
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
}

_REQUEST_SCOPES: ContextVar[frozenset[str]] = ContextVar("baloot_mcp_scopes", default=frozenset({Scope.READ.value}))
_REQUEST_AUTHORIZATION: ContextVar[str | None] = ContextVar("baloot_mcp_authorization", default=None)
_REQUEST_SUBJECT: ContextVar[str] = ContextVar("baloot_mcp_subject", default="")
_REQUEST_TENANT: ContextVar[str] = ContextVar("baloot_mcp_tenant", default="")
_API_CLIENTS: dict[str, httpx.Client] = {}
_API_CLIENT_LOCK = threading.RLock()


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _require_mutations() -> None:
    if not _truthy("BALOOT_MCP_ALLOW_MUTATIONS"):
        raise PermissionError("MCP mutation tools are disabled")
    scopes = _REQUEST_SCOPES.get()
    if Scope.PLAY.value not in scopes and Scope.CONFIG_ADMIN.value not in scopes:
        raise PermissionError("MCP access token is missing baloot.play")


def _require_config_admin() -> None:
    if not _truthy("BALOOT_MCP_ALLOW_MUTATIONS"):
        raise PermissionError("MCP mutation tools are disabled")
    if Scope.CONFIG_ADMIN.value not in _REQUEST_SCOPES.get():
        raise PermissionError("MCP access token is missing baloot.config.admin")


def _require_trusted() -> None:
    scopes = _REQUEST_SCOPES.get()
    if Scope.HIDDEN_READ.value in scopes and _truthy("BALOOT_MCP_ALLOW_TRUSTED_CONTEXT"):
        return
    if os.environ.get("BALOOT_ENV", "development").lower() in {"development", "test"}:
        return
    raise PermissionError("MCP trusted hidden-information tools require baloot.hidden.read")


def _api(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    base = os.environ.get("BALOOT_API_URL", "http://127.0.0.1:8080").rstrip("/")
    headers: dict[str, str] = {"accept": "application/json"}
    delegated = _REQUEST_AUTHORIZATION.get()
    if delegated:
        headers["authorization"] = delegated
    elif os.environ.get("BALOOT_API_KEY"):
        # Service keys remain a local/deployment fallback; user-facing remote
        # MCP requests delegate the verified subject token instead.
        headers["x-api-key"] = os.environ["BALOOT_API_KEY"]
    subject = _REQUEST_SUBJECT.get() or os.environ.get("BALOOT_MCP_SUBJECT_ID", "")
    tenant = _REQUEST_TENANT.get()
    if subject: headers["x-baloot-subject-id"] = subject
    if tenant: headers["x-baloot-tenant-id"] = tenant
    headers["x-baloot-subject-name"] = os.environ.get("BALOOT_MCP_SUBJECT_NAME", "MCP Client")
    if extra_headers:
        headers.update(extra_headers)
    with _API_CLIENT_LOCK:
        client = _API_CLIENTS.get(base)
        if client is None:
            client = httpx.Client(base_url=base, timeout=60.0, limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
            _API_CLIENTS[base] = client
    response = client.request(method, path, json=payload, headers=headers)
    if response.status_code == 409:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if isinstance(detail, dict) and detail.get("reason") == "time_censored":
            raise SearchTimeBudgetExceeded(str(detail.get("message") or "time-censored evidence"))
    response.raise_for_status()
    return response.json()


def _schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


BASE_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_platform_health",
        "description": "Read platform health, version and active service status.",
        "inputSchema": _schema(),
    },
    {
        "name": "get_current_bundle",
        "description": "Read the immutable aggregate configuration bundle used by the engine.",
        "inputSchema": _schema(),
    },
    {
        "name": "get_current_rules",
        "description": "Read the active deterministic Baloot rules profile.",
        "inputSchema": _schema(),
    },
    {
        "name": "list_plugins",
        "description": "List installed and allow-listed platform plugins.",
        "inputSchema": _schema(),
    },
    {
        "name": "get_game",
        "description": "Read public authoritative state for one game.",
        "inputSchema": _schema({"game_id": {"type": "string", "minLength": 1}}, ["game_id"]),
    },
    {
        "name": "validate_context",
        "description": "Validate and legally replay a complete trusted Baloot context.",
        "inputSchema": _schema(
            {"state": {"type": "object"}, "rules_profile": {"type": "string", "default": "full"}},
            ["state"],
        ),
    },
    {
        "name": "analyze_position",
        "description": "Analyze a trusted all-hands context and return observer-safe evidence and move results.",
        "inputSchema": _schema(
            {
                "state": {"type": "object"},
                "observer": {"type": "string", "default": "south"},
                "rules_profile": {"type": "string", "default": "full"},
                "samples": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": BELIEF_POLICY.maximum_particles,
                    "default": 80,
                },
                "exact_allocation_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": (
                        BELIEF_POLICY.maximum_exact_allocation_limit
                    ),
                    "default": 20000,
                },
                "exact_search_remaining": {"type": "integer", "minimum": 0, "default": 12},
                "experimental_conventions": {"type": "boolean", "default": False},
                "seed": {"type": "integer", "minimum": 0, "default": 1},
            },
            ["state"],
        ),
    },
    {
        "name": "create_game",
        "description": "Create an authoritative game session. Mutation tools are disabled by default.",
        "inputSchema": _schema(
            {
                "seed": {"type": "integer", "minimum": 0, "default": 1},
                "dealer": {"type": "string", "default": "east"},
                "rules_profile": {"type": "string", "default": "full"},
            }
        ),
    },
    {
        "name": "submit_bid",
        "description": "Submit one legal bid to an authoritative game.",
        "inputSchema": _schema(
            {
                "game_id": {"type": "string"},
                "seat": {"type": "string"},
                "action": {"type": "string"},
                "trump_suit": {"type": ["string", "null"]},
                "expected_version": {"type": ["integer", "null"]},
            },
            ["game_id", "seat", "action"],
        ),
    },
    {
        "name": "play_card",
        "description": "Play one legal card in an authoritative game.",
        "inputSchema": _schema(
            {
                "game_id": {"type": "string"},
                "seat": {"type": "string"},
                "card": {"type": "string"},
                "ekka": {"type": "boolean", "default": False},
                "expected_version": {"type": ["integer", "null"]},
            },
            ["game_id", "seat", "card"],
        ),
    },
    {
        "name": "create_video",
        "description": (
            "Deprecated compatibility adapter: start a frozen Research V4 round "
            "search. Discover retained results, prepare one, then create a render "
            "batch through the V2 tools. It does not render automatically."
        ),
        "inputSchema": _schema(
            {
                "idempotency_key": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "request": {"type": "object"},
            },
            ["idempotency_key", "request"],
        ),
    },
    {
        "name": "get_video_job",
        "description": (
            "Read an existing stored legacy video job or a V4 compatibility "
            "Research run. Use the V2 Research and render APIs for new work."
        ),
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "cancel_video_job",
        "description": "Request cooperative cancellation of a video job.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "continue_trusted_game",
        "description": "Continue a complete trusted state to the end.",
        "inputSchema": _schema(
            {"state": {"type": "object"}, "rules_profile": {"type": "string", "default": "full"}},
            ["state"],
        ),
    },
)

EXTRA_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_games",
        "description": "List current game sessions and versions.",
        "inputSchema": _schema({"search": {"type": "string", "default": ""}, "limit": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 100}}),
    },
    {
        "name": "get_game_versions",
        "description": "List immutable hash-chained versions for a game.",
        "inputSchema": _schema({"game_id": {"type": "string"}}, ["game_id"]),
    },
    {
        "name": "get_archived_game",
        "description": "Read an immutable archived game by serial, hash, or game ID.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "get_replay_package",
        "description": "Read one immutable game with decisions, audits, verification and video timecodes.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "list_game_decisions",
        "description": "List recorded bidding and play decisions for one game.",
        "inputSchema": _schema({"game_id": {"type": "string"}, "limit": {"type": "integer", "default": 1000}}, ["game_id"]),
    },
    {
        "name": "reanalyze_decision",
        "description": "Create a new versioned analysis for an archived decision.",
        "inputSchema": _schema({"decision_id": {"type": "string"}, "samples": {"type": "integer", "default": 96}, "exact_search_remaining": {"type": "integer", "default": 10}}, ["decision_id"]),
    },
    {
        "name": "add_decision_feedback",
        "description": "Attach expert, referee, or audience feedback to a legal archived decision.",
        "inputSchema": _schema({"decision_id": {"type": "string"}, "preferred_action": {}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}, "source": {"type": "string", "default": "expert"}}, ["decision_id", "preferred_action"]),
    },
    {
        "name": "reopen_game",
        "description": "Reopen an archived game, optionally as an immutable branch.",
        "inputSchema": _schema({"identifier": {"type": "string"}, "branch": {"type": "boolean", "default": True}}, ["identifier"]),
    },
    {
        "name": "fork_decision",
        "description": "Create an immutable counterfactual branch from one archived legal decision and optionally play that branch to the end.",
        "inputSchema": _schema(
            {
                "decision_id": {"type": "string", "minLength": 1},
                "action": {},
                "continue_to_end": {"type": "boolean", "default": True},
                "maximum_actions": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 256},
            },
            ["decision_id", "action"],
        ),
    },
    {
        "name": "list_learning_datasets",
        "description": "List content-addressed ML datasets.",
        "inputSchema": _schema(),
    },
    {
        "name": "create_learning_dataset",
        "description": "Create a grouped, leakage-safe dataset from reviewed decisions.",
        "inputSchema": _schema({"purpose": {"type": "string", "default": "human_corrected_play_v1"}}),
    },
    {
        "name": "list_models",
        "description": "List versioned learning-model candidates and production models.",
        "inputSchema": _schema(),
    },
    {
        "name": "train_model",
        "description": "Train a deterministic policy/value baseline from a registered dataset.",
        "inputSchema": _schema({"dataset_id": {"type": "string"}, "epochs": {"type": "integer", "minimum": 1, "default": 40}}, ["dataset_id"]),
    },
    {
        "name": "mine_challenges",
        "description": "Start progressive challenge discovery using simulation or archived games.",
        "inputSchema": _schema({"request": {"type": "object"}}, ["request"]),
    },
    {
        "name": "list_challenges",
        "description": "List mined and reviewed challenges.",
        "inputSchema": _schema({"status": {"type": ["string", "null"]}, "limit": {"type": "integer", "default": 100}}),
    },
    {
        "name": "start_challenge_publication_review",
        "description": "Move a candidate into its immutable independent-review cycle.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}}, ["challenge_id"]),
    },
    {
        "name": "get_challenge_publication_reviews",
        "description": "Read independent reviews and governed lifecycle events for a challenge.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}}, ["challenge_id"]),
    },
    {
        "name": "submit_challenge_publication_review",
        "description": "Submit one immutable review under the authenticated MCP subject.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}, "decision": {"type": "string", "enum": ["approve", "reject"]}, "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "comment": {"type": "string", "maxLength": 2000}}, ["challenge_id", "decision", "evidence_sha256"]),
    },
    {
        "name": "approve_challenge_publication",
        "description": "Approve a challenge only after all configured independent evidence gates pass.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}, "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, ["challenge_id", "evidence_sha256"]),
    },
    {
        "name": "publish_challenge",
        "description": "Publish an approved challenge with immutable release evidence.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}, "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, ["challenge_id", "evidence_sha256"]),
    },
    {
        "name": "retire_challenge",
        "description": "Retire an approved or published challenge with immutable evidence.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}, "evidence_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, ["challenge_id", "evidence_sha256"]),
    },
    {
        "name": "render_challenge",
        "description": "Render a mined challenge through an allow-listed video plugin.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}, "request": {"type": "object"}}, ["challenge_id"]),
    },
    {
        "name": "create_render_batch",
        "description": "Create an idempotent observer-safe render batch from content-addressed RenderBundleV1 artifacts.",
        "inputSchema": _schema(
            {
                "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 200},
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 100,
                    "items": {"type": "object"},
                },
            },
            ["idempotency_key", "items"],
        ),
    },
    {
        "name": "get_render_batch",
        "description": "Read one governed render-batch job and its evidence.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "cancel_render_batch",
        "description": "Request cooperative cancellation of one governed render batch.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "create_context_search",
        "description": "Create an idempotent critical-context search from a verified trace and reachability proofs.",
        "inputSchema": _schema(
            {
                "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 200},
                "parent_trace_reference": {"type": "object"},
                "decision_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "context_proof_hashes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10000,
                    "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
            ["idempotency_key", "parent_trace_reference", "decision_id", "context_proof_hashes"],
        ),
    },
    {
        "name": "get_context_search",
        "description": "Read one governed critical-context search job and its evidence.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "cancel_context_search",
        "description": "Request cooperative cancellation of one critical-context search.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "create_natural_sakka_generation",
        "description": "Create an idempotent deterministic complete natural Sakka generation job.",
        "inputSchema": _schema({
            "root_seed": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
            "initial_dealer": {"type": "string", "enum": ["north", "east", "south", "west"], "default": "east"},
            "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 200},
        }, ["root_seed", "idempotency_key"]),
    },
    {
        "name": "get_natural_sakka_generation",
        "description": "Read one governed natural Sakka generation job and its immutable result.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "cancel_natural_sakka_generation",
        "description": "Request cooperative cancellation of one natural Sakka generation job.",
        "inputSchema": _schema({"job_id": {"type": "string"}}, ["job_id"]),
    },
    {
        "name": "list_foundry_sakkas",
        "description": "List immutable natural and curated Foundry Sakka artifacts.",
        "inputSchema": _schema({
            "provenance": {"type": ["string", "null"], "enum": ["natural", "curated", None]},
            "after": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
        }),
    },
    {
        "name": "create_natural_sakka",
        "description": "Create one immutable natural Sakka from an uninterrupted ordered trace history.",
        "inputSchema": _schema({
            "trace_references": {"type": "array", "minItems": 1, "maxItems": 1000, "items": {"type": "object"}},
            "target": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 152},
        }, ["trace_references"]),
    },
    {
        "name": "create_curated_sakka",
        "description": "Create one signed curated Sakka from explicit verified contextualized Laffa selections.",
        "inputSchema": _schema({
            "selections": {"type": "array", "minItems": 1, "maxItems": 1000, "items": {"type": "object"}},
            "target": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 152},
        }, ["selections"]),
    },
    {
        "name": "get_foundry_sakka",
        "description": "Load and verify one immutable Foundry Sakka artifact.",
        "inputSchema": _schema({"sakka_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, ["sakka_hash"]),
    },
    {
        "name": "audit_game",
        "description": "Run an observer-safe decision-by-decision audit of an archived game.",
        "inputSchema": _schema({"identifier": {"type": "string"}, "maximum_decisions": {"type": "integer", "default": 12}, "rerun": {"type": "boolean", "default": False}}, ["identifier"]),
    },
    {
        "name": "explain_decision",
        "description": "Explain one recorded decision, alternatives, regret, facts, hypotheses and recorded intent.",
        "inputSchema": _schema({"decision_id": {"type": "string"}, "rerun": {"type": "boolean", "default": False}, "samples": {"type": "integer", "default": 96}}, ["decision_id"]),
    },
    {
        "name": "compare_decision_versions",
        "description": "Compare all engine analyses, audits and expert feedback for one decision.",
        "inputSchema": _schema({"decision_id": {"type": "string"}}, ["decision_id"]),
    },
    {
        "name": "record_challenge_attempt",
        "description": "Record one player answer and recalibrate challenge difficulty.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}, "chosen_action": {}, "player_id": {"type": ["string", "null"]}, "response_seconds": {"type": ["number", "null"]}, "confidence": {"type": ["number", "null"]}}, ["challenge_id", "chosen_action"]),
    },
    {
        "name": "get_challenge_difficulty",
        "description": "Read engine and audience-calibrated difficulty for a challenge.",
        "inputSchema": _schema({"challenge_id": {"type": "string"}}, ["challenge_id"]),
    },
    {
        "name": "record_decision_attempt",
        "description": "Record a replay-studio answer and update audience-calibrated decision difficulty.",
        "inputSchema": _schema({"decision_id": {"type": "string"}, "chosen_action": {}, "player_id": {"type": ["string", "null"]}, "response_seconds": {"type": ["number", "null"]}, "confidence": {"type": ["number", "null"]}}, ["decision_id", "chosen_action"]),
    },
    {
        "name": "get_decision_difficulty",
        "description": "Read engine and audience-calibrated difficulty for an archived decision.",
        "inputSchema": _schema({"decision_id": {"type": "string"}}, ["decision_id"]),
    },
    {
        "name": "get_video_record",
        "description": "Read one immutable video record, provenance manifest, and linked decisions.",
        "inputSchema": _schema({"video_id": {"type": "string"}}, ["video_id"]),
    },
    {
        "name": "get_video_timeline",
        "description": "Read a video's complete segment and decision timecode timeline.",
        "inputSchema": _schema({"video_id": {"type": "string"}}, ["video_id"]),
    },
    {
        "name": "get_video_audit_map",
        "description": "Read a video's decision/timecode provenance map.",
        "inputSchema": _schema({"video_id": {"type": "string"}}, ["video_id"]),
    },
    {
        "name": "render_game_replay",
        "description": "Render an immutable archived game as a full historical replay video with every bid/play decision mapped to a timecode. Mutation tools are disabled by default.",
        "inputSchema": _schema(
            {
                "identifier": {"type": "string", "minLength": 1},
                "decision_id": {"type": ["string", "null"]},
                "renderer_plugin": {"type": "string", "default": "premium_video"},
                "theme": {"type": "string", "default": "emerald"},
                "width": {"type": "integer", "minimum": 720, "maximum": 2160, "default": 1080},
                "height": {"type": "integer", "minimum": 1280, "maximum": 3840, "default": 1920},
                "fps": {"type": "integer", "minimum": 24, "maximum": 60, "default": 30},
                "crf": {"type": "integer", "minimum": 14, "maximum": 30, "default": 18},
                "audit_samples": {"type": "integer", "minimum": 4, "maximum": 4096, "default": 96},
            },
            ["identifier"],
        ),
    },
    {
        "name": "annotate_player_intent",
        "description": "Record a player or expert statement about why a legal archived decision was made.",
        "inputSchema": _schema({"decision_id": {"type": "string"}, "selected_action": {}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "rationale": {"type": "string"}, "player_id": {"type": ["string", "null"]}}, ["decision_id", "selected_action", "rationale"]),
    },
    {
        "name": "adaptive_next_challenge",
        "description": "Choose the next published challenge near the player's calibrated skill while avoiding recent items.",
        "inputSchema": _schema({"player_id": {"type": ["string", "null"]}, "recent_ids": {"type": "array", "items": {"type": "string"}, "default": []}}),
    },
)

SAKKA_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_sakka_boards",
        "description": "List same-start Sakka boards visible to the current API/MCP subject.",
        "inputSchema": _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}}),
    },
    {
        "name": "verify_sakka_board",
        "description": "Verify the immutable Sakka board hash, trusted payload and server signature.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "create_sakka_board",
        "description": "Create an immutable same-start Sakka from a new deal or an existing game.",
        "inputSchema": _schema({"request": {"type": "object"}}, ["request"]),
    },
    {
        "name": "create_sakka_room",
        "description": "Create a shareable Sakka room and human-readable code.",
        "inputSchema": _schema({"board_identifier": {"type": "string"}, "mode": {"type": "string", "default": "async"}, "max_players": {"type": "integer", "default": 16}, "allow_guests": {"type": "boolean", "default": True}}, ["board_identifier"]),
    },
    {
        "name": "get_sakka_room",
        "description": "Read one Sakka room, members and leaderboard by room ID or share code.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "join_sakka_room",
        "description": "Join a Sakka room as the configured MCP/API subject.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "start_sakka_run",
        "description": "Start or reopen the current subject's run from exactly the shared Sakka state.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "verify_sakka_run",
        "description": "Replay and verify the complete tamper-evident Sakka action chain.",
        "inputSchema": _schema({"run_id": {"type": "string"}}, ["run_id"]),
    },
    {
        "name": "play_sakka_action",
        "description": "Submit one legal bid or card action in a Sakka run.",
        "inputSchema": _schema({"run_id": {"type": "string"}, "action": {"type": "object"}}, ["run_id", "action"]),
    },
    {
        "name": "reopen_sakka_run",
        "description": "Reopen the current subject's saved Sakka run.",
        "inputSchema": _schema({"run_id": {"type": "string"}}, ["run_id"]),
    },
    {
        "name": "get_sakka_leaderboard",
        "description": "Read the fair-comparison leaderboard for a Sakka room.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "compare_sakka_runs",
        "description": "Verify and compare two completed runs from the same immutable Sakka board.",
        "inputSchema": _schema({
            "run_a": {"type": "string", "minLength": 1},
            "run_b": {"type": "string", "minLength": 1},
        }, ["run_a", "run_b"]),
    },
    {
        "name": "audit_sakka_run",
        "description": "Analyze the current subject's Sakka decisions and compare all legal alternatives.",
        "inputSchema": _schema({
            "run_id": {"type": "string"}, "deep": {"type": "boolean", "default": False},
            "force": {"type": "boolean", "default": False},
            "maximum_decisions": {"type": "integer", "minimum": 1, "maximum": 256, "default": 32},
        }, ["run_id"]),
    },
    {
        "name": "share_sakka_decision",
        "description": "Create a new immutable Sakka beginning at one action in the current subject's run.",
        "inputSchema": _schema({
            "run_id": {"type": "string"}, "sequence": {"type": "integer", "minimum": 0},
            "scope": {"type": "string", "enum": ["round", "match"], "default": "round"},
            "label": {"type": ["string", "null"]},
        }, ["run_id", "sequence"]),
    },
    {
        "name": "review_sakka_decision",
        "description": "Record player or expert feedback on one Sakka decision for future training.",
        "inputSchema": _schema({
            "run_id": {"type": "string"}, "sequence": {"type": "integer", "minimum": 0},
            "preferred_action": {}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string", "minLength": 1},
            "interestingness": {"type": "number", "minimum": 0, "maximum": 1, "default": 0},
        }, ["run_id", "sequence", "preferred_action", "rationale"]),
    },
)

SHARED_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_shared_boards",
        "description": "List standardized shared لفة and صكّة boards visible to the current subject.",
        "inputSchema": _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}}),
    },
    {
        "name": "verify_shared_board",
        "description": "Verify a shared board's immutable environment, cards, agent tuning and commitments.",
        "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"]),
    },
    {
        "name": "create_shared_board",
        "description": "Create one standardized shared لفة or complete صكّة. Mutations are disabled by default.",
        "inputSchema": _schema({"request": {"type": "object"}}, ["request"]),
    },
    {
        "name": "create_shared_room",
        "description": "Create a room from an immutable shared board.",
        "inputSchema": _schema({"board_identifier": {"type": "string"}, "mode": {"type": "string", "default": "async"}, "max_players": {"type": "integer", "default": 16}, "allow_guests": {"type": "boolean", "default": True}, "settings": {"type": "object", "default": {}}}, ["board_identifier"]),
    },
    {"name": "get_shared_room", "description": "Read a shared room by ID or code.", "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"])},
    {"name": "join_shared_room", "description": "Join a shared room as the current subject.", "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"])},
    {"name": "start_shared_run", "description": "Start or reopen the current subject's private run from the exact standardized environment.", "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"])},
    {"name": "reopen_shared_run", "description": "Reopen the current subject's saved shared run.", "inputSchema": _schema({"run_id": {"type": "string"}}, ["run_id"])},
    {"name": "verify_shared_run", "description": "Verify the run action chain and deterministic replay.", "inputSchema": _schema({"run_id": {"type": "string"}}, ["run_id"])},
    {"name": "play_shared_action", "description": "Submit one legal bid or card action in a shared run.", "inputSchema": _schema({"run_id": {"type": "string"}, "action": {"type": "object"}}, ["run_id", "action"])},
    {"name": "get_shared_run_review", "description": "Read all bid/play decisions and simple Arabic explanations for one owned run.", "inputSchema": _schema({"run_id": {"type": "string"}, "ensure_audit": {"type": "boolean", "default": False}}, ["run_id"])},
    {"name": "explain_shared_decision", "description": "Explain one completed action in simple attributed Arabic without leaking challenge hints.", "inputSchema": _schema({"run_id": {"type": "string"}, "decision_id": {"type": "string"}, "level": {"type": "string", "enum": ["basic", "coach", "expert"], "default": "basic"}, "surface": {"type": "string", "enum": ["active", "replay", "video", "challenge"], "default": "replay"}, "challenge_revealed": {"type": "boolean", "default": False}, "ensure_audit": {"type": "boolean", "default": False}, "view_mode": {"type": "string", "enum": ["observer", "public", "oracle_after_completion"], "default": "observer"}}, ["run_id", "decision_id"])},
    {"name": "get_shared_decision_reviews", "description": "Read structured feedback and training consensus for one decision.", "inputSchema": _schema({"run_id": {"type": "string"}, "decision_id": {"type": "string"}}, ["run_id", "decision_id"])},
    {"name": "review_shared_decision", "description": "Submit a legal better-play proposal with reasons. One review never becomes truth by itself.", "inputSchema": _schema({"run_id": {"type": "string"}, "decision_id": {"type": "string"}, "preferred_action": {}, "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8}, "reason_codes": {"type": "array", "items": {"type": "string"}}, "rationale": {"type": "string", "minLength": 3}, "evidence": {"type": "array", "items": {"type": "object"}}, "view_mode": {"type": "string", "default": "observer"}}, ["run_id", "decision_id", "preferred_action", "rationale"])},
    {"name": "share_shared_decision", "description": "Create a new shared لفة or صكّة from one action in an owned run.", "inputSchema": _schema({"run_id": {"type": "string"}, "sequence": {"type": "integer", "minimum": 0}, "scope": {"type": "string", "enum": ["laffa", "sakka"], "default": "laffa"}, "maximum_rounds": {"type": "integer", "minimum": 1, "maximum": 64, "default": 16}, "label": {"type": ["string", "null"]}}, ["run_id", "sequence"])},
    {"name": "get_shared_leaderboard", "description": "Read the standardized leaderboard for one shared room.", "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"])},
    {"name": "compare_shared_runs", "description": "Compare two compatible runs from the same frozen environment.", "inputSchema": _schema({"run_a": {"type": "string"}, "run_b": {"type": "string"}}, ["run_a", "run_b"])},
)

V17_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"name": "list_live_rooms", "description": "List synchronous four-human Baloot rooms visible to the current subject.", "inputSchema": _schema({"status": {"type": ["string", "null"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}})},
    {"name": "create_live_room", "description": "Create a synchronous four-human Baloot room.", "inputSchema": _schema({"request": {"type": "object"}}, ["request"])},
    {"name": "get_live_room", "description": "Read public room/game state for a synchronous table.", "inputSchema": _schema({"identifier": {"type": "string"}, "since_sequence": {"type": "integer", "minimum": 0, "default": 0}}, ["identifier"])},
    {"name": "claim_live_seat", "description": "Claim one seat in a waiting synchronous table.", "inputSchema": _schema({"identifier": {"type": "string"}, "seat": {"type": "string"}, "display_name": {"type": "string"}}, ["identifier", "seat", "display_name"])},
    {"name": "set_live_ready", "description": "Set the current subject ready or not ready.", "inputSchema": _schema({"identifier": {"type": "string"}, "ready": {"type": "boolean", "default": True}}, ["identifier"])},
    {"name": "start_live_table", "description": "Start a four-human table after all seats are ready.", "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"])},
    {"name": "play_live_action", "description": "Submit one legal bid or card action for the current subject's seat.", "inputSchema": _schema({"identifier": {"type": "string"}, "action": {"type": "object"}}, ["identifier", "action"])},
    {"name": "verify_live_room", "description": "Verify live room event hash chain and room state hash.", "inputSchema": _schema({"identifier": {"type": "string"}}, ["identifier"])},
    {"name": "list_adjudication_cases", "description": "List documented adjudication cases.", "inputSchema": _schema({"game_id": {"type": ["string", "null"]}, "status": {"type": ["string", "null"]}, "limit": {"type": "integer", "default": 100}})},
    {"name": "open_adjudication_case", "description": "Open a tamper-evident rule/adjudication case.", "inputSchema": _schema({"request": {"type": "object"}}, ["request"])},
    {"name": "get_adjudication_case", "description": "Read one adjudication case.", "inputSchema": _schema({"case_id": {"type": "string"}}, ["case_id"])},
    {"name": "recommend_adjudication_ruling", "description": "Read the configured non-binding ruling recommendation for a case.", "inputSchema": _schema({"case_id": {"type": "string"}}, ["case_id"])},
    {"name": "get_security_capabilities", "description": "Read optional sign-in, OIDC, passkey and role/scope capabilities.", "inputSchema": _schema()},
)


V18_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"name":"get_hitl_profile","description":"Read governed Human-in-the-Loop review and consent policy.","inputSchema":_schema()},
    {"name":"list_hitl_cases","description":"List governed review cases.","inputSchema":_schema({"limit":{"type":"integer","minimum":1,"maximum":500,"default":100}})},
    {"name":"create_hitl_case","description":"Open a governed review case from an archived decision.","inputSchema":_schema({"decision_id":{"type":"string"},"correction_type":{"type":"string","default":"analysis_defect"},"high_impact":{"type":"boolean","default":False}},["decision_id"])},
    {"name":"get_hitl_case","description":"Read a review case; blind mode hides production recommendation and peer feedback.","inputSchema":_schema({"case_id":{"type":"string"},"blind":{"type":"boolean","default":False}},["case_id"])},
    {"name":"assign_hitl_review","description":"Start a blind review assignment for the current subject.","inputSchema":_schema({"case_id":{"type":"string"},"device_id":{"type":"string","default":"mcp"},"network_id":{"type":"string","default":"mcp"}},["case_id"])},
    {"name":"submit_hitl_review","description":"Submit a legal alternative with structured reasons and disclosure state.","inputSchema":_schema({"case_id":{"type":"string"},"assignment_id":{"type":"string"},"assignment_token":{"type":"string"},"preferred_action":{},"confidence":{"type":"number","minimum":0,"maximum":1},"reason_codes":{"type":"array","items":{"type":"string"}},"rationale_ar":{"type":"string","minLength":3},"claims":{"type":"array","items":{"type":"object"}},"evidence":{"type":"array","items":{"type":"object"}},"hidden_cards_seen":{"type":"boolean","default":False},"final_result_seen":{"type":"boolean","default":False}},["case_id","assignment_id","assignment_token","preferred_action","rationale_ar"])},
    {"name":"compute_hitl_consensus","description":"Compute independent weighted consensus with confidence bounds and moderation gates.","inputSchema":_schema({"case_id":{"type":"string"}},["case_id"])},
    {"name":"set_hitl_consent","description":"Set revocable training, research, public challenge, voice and attribution consent.","inputSchema":_schema({"allow_training":{"type":"boolean","default":False},"allow_research":{"type":"boolean","default":False},"allow_public_challenge":{"type":"boolean","default":False},"allow_voice":{"type":"boolean","default":False},"allow_identity_attribution":{"type":"boolean","default":False},"allow_private_coaching":{"type":"boolean","default":True},"revoke":{"type":"boolean","default":False}})},
    {"name":"create_hitl_training_candidate","description":"Create a governed training candidate; never promotes a production model automatically.","inputSchema":_schema({"case_id":{"type":"string"}},["case_id"])},
    {"name":"record_hitl_challenger_evaluation","description":"Record one independent challenger evaluation for a governed candidate.","inputSchema":_schema({"candidate_id":{"type":"string"},"validator_id":{"type":"string"},"validator_type":{"type":"string"},"status":{"type":"string"},"metrics":{"type":"object"},"independent":{"type":"boolean","default":True}},["candidate_id","validator_id","status"])},
    {"name":"create_engine_improvement_proposal","description":"Propose a governed engine/model/search/belief/convention/rule change. It never auto-applies.","inputSchema":_schema({"target":{"type":"string"},"title_ar":{"type":"string"},"rationale_ar":{"type":"string"},"source_case_ids":{"type":"array","items":{"type":"string"}},"requested_change":{"type":"object"}},["target","title_ar","rationale_ar"])},
    {"name":"evaluate_engine_improvement","description":"Record independent evidence for an improvement proposal.","inputSchema":_schema({"proposal_id":{"type":"string"},"validator_id":{"type":"string"},"status":{"type":"string"},"evidence":{"type":"object"},"independent":{"type":"boolean","default":True}},["proposal_id","validator_id","status"])},
    {"name":"render_hitl_review_clip","description":"Render an observer-safe review clip with no recommendation or crowd consensus before submission.","inputSchema":_schema({"case_id":{"type":"string"},"width":{"type":"integer","minimum":320},"height":{"type":"integer","minimum":320},"fps":{"type":"integer"}},["case_id"])},
)

V19_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"name":"health_check","description":"Read platform health and active version.","inputSchema":_schema()},
    {"name":"list_platform_capabilities","description":"List the canonical REST/gRPC/MCP capability matrix.","inputSchema":_schema()},
    {"name":"get_configuration_capabilities","description":"Read the row-versioned configuration control-plane contract.","inputSchema":_schema()},
    {"name":"list_configuration_profiles","description":"List tenant-aware versioned configuration profiles.","inputSchema":_schema({"family":{"type":["string","null"]},"profile_id":{"type":["string","null"]},"status":{"type":["string","null"]},"limit":{"type":"integer","minimum":1,"maximum":1000,"default":200}})},
    {"name":"create_configuration_profile","description":"Create an immutable configuration profile draft.","inputSchema":_schema({"family":{"type":"string"},"payload":{"type":"object"},"scope":{"type":"string","default":"global"},"tenant_id":{"type":["string","null"]}},["family","payload"])},
    {"name":"publish_configuration_profile","description":"Publish a validated configuration profile draft.","inputSchema":_schema({"row_id":{"type":"string"},"expected_etag":{"type":"integer","minimum":1}},["row_id","expected_etag"])},
    {"name":"reject_configuration_profile","description":"Reject a configuration profile draft with an auditable reason.","inputSchema":_schema({"row_id":{"type":"string"},"expected_etag":{"type":"integer","minimum":1},"reason":{"type":"string"}},["row_id","expected_etag","reason"])},
    {"name":"retire_configuration_profile","description":"Retire a published configuration profile with an auditable reason.","inputSchema":_schema({"row_id":{"type":"string"},"expected_etag":{"type":"integer","minimum":1},"reason":{"type":"string"}},["row_id","expected_etag","reason"])},
    {"name":"resolve_configuration_profile","description":"Resolve tenant then global published configuration.","inputSchema":_schema({"family":{"type":"string"},"profile_id":{"type":"string"}},["family","profile_id"])},
    {"name":"compare_configuration_profiles","description":"Return a structured path diff between two profile versions.","inputSchema":_schema({"left":{"type":"string"},"right":{"type":"string"}},["left","right"])},
    {"name":"list_configuration_bundles","description":"List versioned configuration bundles.","inputSchema":_schema({"bundle_id":{"type":["string","null"]},"status":{"type":["string","null"]},"limit":{"type":"integer","minimum":1,"maximum":1000,"default":200}})},
    {"name":"create_configuration_bundle","description":"Create a configuration bundle draft from profile revision IDs.","inputSchema":_schema({"bundle_id":{"type":"string"},"version":{"type":"string"},"components":{"type":"object","additionalProperties":{"type":"string"}},"mode":{"type":"string","default":"full"},"scope":{"type":"string","default":"global"},"tenant_id":{"type":["string","null"]}},["bundle_id","version","components"])},
    {"name":"publish_configuration_bundle","description":"Publish a fully validated configuration bundle.","inputSchema":_schema({"row_id":{"type":"string"},"expected_etag":{"type":"integer","minimum":1}},["row_id","expected_etag"])},
    {"name":"reject_configuration_bundle","description":"Reject a configuration bundle draft with an auditable reason.","inputSchema":_schema({"row_id":{"type":"string"},"expected_etag":{"type":"integer","minimum":1},"reason":{"type":"string"}},["row_id","expected_etag","reason"])},
    {"name":"retire_configuration_bundle","description":"Retire a published configuration bundle with an auditable reason.","inputSchema":_schema({"row_id":{"type":"string"},"expected_etag":{"type":"integer","minimum":1},"reason":{"type":"string"}},["row_id","expected_etag","reason"])},
    {"name":"resolve_configuration_bundle","description":"Resolve the active published configuration bundle.","inputSchema":_schema({"bundle_id":{"type":"string"},"expanded":{"type":"boolean","default":False}},["bundle_id"])},
    {"name":"resolve_effective_runtime_config_v2","description":"Resolve the immutable, content-addressed V2 runtime configuration contract.","inputSchema":_schema({"bundle_id":{"type":"string"}},["bundle_id"])},
    {"name":"rollback_configuration_bundle","description":"Create and publish a new immutable bundle revision from a historical published revision.","inputSchema":_schema({"revision_id":{"type":"string"},"new_version":{"type":"string"}},["revision_id","new_version"])},
    {"name":"get_performance_profile","description":"Read performance budgets and observability policy.","inputSchema":_schema()},
    {"name":"run_performance_quick","description":"Run bounded control-plane performance probes.","inputSchema":_schema()},
    {"name":"get_certification_profile","description":"Read the rules-certification governance profile.","inputSchema":_schema()},
    {"name":"verify_certification_candidate","description":"Verify a signed rules-certification candidate without activating rules.","inputSchema":_schema({"candidate":{"type":"object"}},["candidate"])},
    {"name":"get_benchmark_profile","description":"Read the intelligence benchmark and promotion-gate profile.","inputSchema":_schema()},
    {"name":"evaluate_intelligence_benchmark","description":"Evaluate supplied model decisions; never promotes a model automatically.","inputSchema":_schema({"model_id":{"type":"string"},"examples":{"type":"array","items":{"type":"object"}}},["model_id","examples"])},
    {"name":"get_operations_profile","description":"Read SLO and recovery objectives.","inputSchema":_schema()},
)

def _tool_annotations(definition: dict[str, Any]) -> dict[str, Any]:
    value = dict(definition); name = str(value.get("name") or "")
    mutation_prefixes = ("create_", "submit_", "play_", "set_", "assign_", "record_", "render_", "approve_", "publish_", "retire_", "reject_", "cancel_", "open_", "claim_", "start_")
    destructive = name.startswith(("delete_", "retire_", "reject_", "cancel_"))
    read_only = not name.startswith(mutation_prefixes) and not destructive
    value["annotations"] = {
        "readOnlyHint": read_only, "destructiveHint": destructive,
        "idempotentHint": read_only or name.startswith(("get_", "list_", "verify_", "resolve_", "compare_")),
        "openWorldHint": name.startswith(("render_", "run_", "create_")),
    }
    return value

TOOL_DEFINITIONS = tuple(_tool_annotations(item) for item in (BASE_TOOL_DEFINITIONS + EXTRA_TOOL_DEFINITIONS + SHARED_TOOL_DEFINITIONS + SAKKA_TOOL_DEFINITIONS + V17_TOOL_DEFINITIONS + V18_TOOL_DEFINITIONS + V19_TOOL_DEFINITIONS))



def _require_argument(arguments: dict[str, Any], name: str) -> Any:
    if name not in arguments:
        raise MCPError(-32602, f"missing required argument: {name}")
    return arguments[name]


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "get_platform_health":
        return _api("GET", "/healthz")
    if name == "health_check":
        return _api("GET", "/api/v1/health")
    if name == "list_platform_capabilities":
        return _api("GET", "/api/v1/platform/capabilities")
    if name == "get_configuration_capabilities":
        return _api("GET", "/api/v1/configuration-registry/profile")
    if name == "list_configuration_profiles":
        query=[]
        for source,target in (("family","family"),("profile_id","profileId"),("status","status"),("limit","limit")):
            if arguments.get(source) is not None: query.append(f"{target}={arguments[source]}")
        return _api("GET", "/api/v1/configuration-registry/profiles" + (("?"+"&".join(query)) if query else ""))
    if name == "create_configuration_profile":
        _require_config_admin()
        return _api("POST", "/api/v1/configuration-registry/profiles", {"family":arguments["family"],"payload":arguments["payload"],"scope":arguments.get("scope","global"),"tenantId":arguments.get("tenant_id")})
    if name == "publish_configuration_profile":
        _require_config_admin(); return _api("POST", f"/api/v1/configuration-registry/profiles/{arguments['row_id']}:publish", {"expectedEtag":arguments["expected_etag"]})
    if name == "reject_configuration_profile":
        _require_config_admin(); return _api("POST", f"/api/v1/configuration-registry/profiles/{arguments['row_id']}:reject", {"expectedEtag":arguments["expected_etag"],"reason":arguments["reason"]})
    if name == "retire_configuration_profile":
        _require_config_admin(); return _api("POST", f"/api/v1/configuration-registry/profiles/{arguments['row_id']}:retire", {"expectedEtag":arguments["expected_etag"],"reason":arguments["reason"]})
    if name == "resolve_configuration_profile":
        return _api("GET", f"/api/v1/configuration-registry/profiles:resolve?family={arguments['family']}&profileId={arguments['profile_id']}")
    if name == "compare_configuration_profiles":
        return _api("GET", f"/api/v1/configuration-registry/profiles:compare?left={arguments['left']}&right={arguments['right']}")
    if name == "list_configuration_bundles":
        query=[]
        for source,target in (("bundle_id","bundleId"),("status","status"),("limit","limit")):
            if arguments.get(source) is not None: query.append(f"{target}={arguments[source]}")
        return _api("GET", "/api/v1/configuration-registry/bundles" + (("?"+"&".join(query)) if query else ""))
    if name == "create_configuration_bundle":
        _require_config_admin()
        return _api("POST", "/api/v1/configuration-registry/bundles", {"bundleId":arguments["bundle_id"],"version":arguments["version"],"mode":arguments.get("mode","full"),"componentRevisions":arguments["components"],"scope":arguments.get("scope","global"),"tenantId":arguments.get("tenant_id")})
    if name == "publish_configuration_bundle":
        _require_config_admin(); return _api("POST", f"/api/v1/configuration-registry/bundles/{arguments['row_id']}:publish", {"expectedEtag":arguments["expected_etag"]})
    if name == "reject_configuration_bundle":
        _require_config_admin(); return _api("POST", f"/api/v1/configuration-registry/bundles/{arguments['row_id']}:reject", {"expectedEtag":arguments["expected_etag"],"reason":arguments["reason"]})
    if name == "retire_configuration_bundle":
        _require_config_admin(); return _api("POST", f"/api/v1/configuration-registry/bundles/{arguments['row_id']}:retire", {"expectedEtag":arguments["expected_etag"],"reason":arguments["reason"]})
    if name == "resolve_configuration_bundle":
        expanded="true" if arguments.get("expanded",False) else "false"
        return _api("GET", f"/api/v1/configuration-registry/resolve?bundleId={arguments['bundle_id']}")
    if name == "resolve_effective_runtime_config_v2":
        return _api("GET", f"/api/v1/configuration-registry/resolve-v2?bundleId={arguments['bundle_id']}")
    if name == "rollback_configuration_bundle":
        _require_config_admin()
        return _api(
            "POST",
            f"/api/v1/configuration-registry/bundles/{arguments['revision_id']}:rollback",
            {"newVersion": arguments["new_version"]},
        )
    if name == "get_performance_profile":
        return _api("GET", "/api/v1/performance/profile")
    if name == "run_performance_quick":
        _require_mutations(); return _api("POST", "/api/v1/performance/quick", {})
    if name == "get_certification_profile":
        return _api("GET", "/api/v1/certification/profile")
    if name == "verify_certification_candidate":
        _require_config_admin(); return _api("POST", "/api/v1/certification/candidates:verify", {"candidate":arguments["candidate"]})
    if name == "get_benchmark_profile":
        return _api("GET", "/api/v1/benchmark/profile")
    if name == "evaluate_intelligence_benchmark":
        _require_config_admin(); return _api("POST", "/api/v1/benchmark/evaluations", {"modelId":arguments["model_id"],"examples":arguments["examples"]})
    if name == "get_operations_profile":
        return _api("GET", "/api/v1/operations/profile")
    if name == "get_current_bundle":
        return build_bundle(ROOT).to_dict()
    if name == "get_current_rules":
        return build_bundle(ROOT).rules.data
    if name == "list_plugins":
        return _api("GET", "/api/v1/plugins")
    if name == "get_game":
        return _api("GET", f"/api/v1/games/{_require_argument(arguments, 'game_id')}")
    if name == "validate_context":
        _require_trusted()
        bundle = build_bundle(ROOT, mode=str(arguments.get("rules_profile", "full")))
        replayed = replay_trusted_context(_require_argument(arguments, "state"), bundle.rules)
        return {"valid": True, "bundleHash": bundle.sha256, "state": replayed.to_dict(reveal_hands=True)}
    if name == "analyze_position":
        _require_trusted()
        bundle = build_bundle(ROOT, mode=str(arguments.get("rules_profile", "full")))
        complete = replay_trusted_context(_require_argument(arguments, "state"), bundle.rules)
        seat = Seat.parse(str(arguments.get("observer", "south")))
        if complete.current_player != seat:
            raise ValueError("observer must be the current player")
        information = InformationState.from_complete(complete, seat)
        conventions = json.loads(json.dumps(bundle.convention))
        conventions["apply_to_probability"] = bool(arguments.get("experimental_conventions", False))
        result = analyze(
            information,
            bundle.rules,
            conventions,
            requested_samples=int(arguments.get("samples", 80)),
            exact_allocation_limit=int(arguments.get("exact_allocation_limit", 20000)),
            exact_search_max_remaining=int(arguments.get("exact_search_remaining", 12)),
            seed=int(arguments.get("seed", 1)),
        )
        return {"bundleHash": bundle.sha256, "informationState": information.to_dict(), "analysis": result.to_dict()}
    if name == "create_game":
        _require_mutations()
        return _api(
            "POST",
            "/api/v1/games",
            {
                "seed": int(arguments.get("seed", 1)),
                "dealer": str(arguments.get("dealer", "east")),
                "rulesProfile": str(arguments.get("rules_profile", "full")),
            },
        )
    if name == "submit_bid":
        _require_mutations()
        return _api(
            "POST",
            f"/api/v1/games/{_require_argument(arguments, 'game_id')}/bids",
            {
                "seat": _require_argument(arguments, "seat"),
                "action": _require_argument(arguments, "action"),
                "trumpSuit": arguments.get("trump_suit"),
                "expectedVersion": arguments.get("expected_version"),
            },
        )
    if name == "play_card":
        _require_mutations()
        return _api(
            "POST",
            f"/api/v1/games/{_require_argument(arguments, 'game_id')}/actions",
            {
                "seat": _require_argument(arguments, "seat"),
                "card": _require_argument(arguments, "card"),
                "ekka": bool(arguments.get("ekka", False)),
                "expectedVersion": arguments.get("expected_version"),
            },
        )
    if name == "create_video":
        _require_mutations()
        _require_trusted()
        return _api(
            "POST",
            "/api/v1/video-jobs",
            _require_argument(arguments, "request"),
            extra_headers={
                "idempotency-key": str(
                    _require_argument(arguments, "idempotency_key")
                )
            },
        )
    if name == "get_video_job":
        return _api("GET", f"/api/v1/video-jobs/{_require_argument(arguments, 'job_id')}")
    if name == "cancel_video_job":
        _require_mutations()
        return _api("POST", f"/api/v1/video-jobs/{_require_argument(arguments, 'job_id')}:cancel", {})
    if name == "continue_trusted_game":
        _require_mutations()
        _require_trusted()
        bundle = build_bundle(ROOT, mode=str(arguments.get("rules_profile", "full")))
        final = continue_trusted_context(_require_argument(arguments, "state"), bundle.rules)
        return {"bundleHash": bundle.sha256, "state": final.to_dict(reveal_hands=True)}
    if name == "list_games":
        query = str(arguments.get("search", ""))
        limit = int(arguments.get("limit", 100))
        return _api("GET", f"/api/v1/games?search={query}&limit={limit}")
    if name == "get_game_versions":
        return _api("GET", f"/api/v1/games/{_require_argument(arguments, 'game_id')}/versions")
    if name == "get_archived_game":
        return _api("GET", f"/api/v1/history/games/{_require_argument(arguments, 'identifier')}")
    if name == "get_replay_package":
        return _api("GET", f"/api/v1/history/games/{_require_argument(arguments, 'identifier')}/replay")
    if name == "list_game_decisions":
        game_id = _require_argument(arguments, "game_id")
        return _api("GET", f"/api/v1/games/{game_id}/decisions?limit={int(arguments.get('limit', 1000))}")
    if name == "reanalyze_decision":
        _require_mutations()
        decision_id = _require_argument(arguments, "decision_id")
        return _api("POST", f"/api/v1/history/decisions/{decision_id}:reanalyze", {
            "samples": int(arguments.get("samples", 96)),
            "exactSearchRemaining": int(arguments.get("exact_search_remaining", 10)),
        })
    if name == "add_decision_feedback":
        _require_mutations()
        decision_id = _require_argument(arguments, "decision_id")
        return _api("POST", f"/api/v1/history/decisions/{decision_id}/feedback", {
            "preferredAction": _require_argument(arguments, "preferred_action"),
            "confidence": float(arguments.get("confidence", 0.8)),
            "rationale": str(arguments.get("rationale", "")),
            "source": str(arguments.get("source", "expert")),
        })
    if name == "reopen_game":
        _require_mutations()
        identifier = _require_argument(arguments, "identifier")
        branch = "true" if bool(arguments.get("branch", True)) else "false"
        return _api("POST", f"/api/v1/history/games/{identifier}:reopen?branch={branch}", {})
    if name == "fork_decision":
        _require_mutations()
        decision_id = _require_argument(arguments, "decision_id")
        return _api("POST", f"/api/v1/history/decisions/{decision_id}:fork", {
            "action": _require_argument(arguments, "action"),
            "continueToEnd": bool(arguments.get("continue_to_end", True)),
            "maximumActions": int(arguments.get("maximum_actions", 256)),
        })
    if name == "list_learning_datasets":
        return _api("GET", "/api/v1/learning/datasets")
    if name == "create_learning_dataset":
        _require_mutations()
        return _api("POST", "/api/v1/learning/datasets", {"purpose": str(arguments.get("purpose", "human_corrected_play_v1"))})
    if name == "list_models":
        return _api("GET", "/api/v1/learning/models")
    if name == "train_model":
        _require_mutations()
        return _api("POST", "/api/v1/learning/models:train", {"datasetId": _require_argument(arguments, "dataset_id"), "epochs": int(arguments.get("epochs", 40))})
    if name == "mine_challenges":
        _require_mutations()
        return _api("POST", "/api/v1/challenge-jobs", _require_argument(arguments, "request"))
    if name == "list_challenges":
        status = arguments.get("status")
        path = f"/api/v1/challenges?limit={int(arguments.get('limit', 100))}"
        if status:
            path += f"&status={status}"
        return _api("GET", path)
    if name == "start_challenge_publication_review":
        _require_mutations()
        return _api("POST", f"/api/v1/challenges/{_require_argument(arguments, 'challenge_id')}:status", {"status": "review"})
    if name == "get_challenge_publication_reviews":
        return _api("GET", f"/api/v1/challenges/{_require_argument(arguments, 'challenge_id')}/publication-reviews")
    if name == "submit_challenge_publication_review":
        _require_mutations()
        return _api("POST", f"/api/v1/challenges/{_require_argument(arguments, 'challenge_id')}/publication-reviews", {
            "decision": _require_argument(arguments, "decision"),
            "evidenceSha256": _require_argument(arguments, "evidence_sha256"),
            "comment": str(arguments.get("comment", "")),
        })
    if name in {"approve_challenge_publication", "publish_challenge", "retire_challenge"}:
        _require_config_admin()
        suffix = {"approve_challenge_publication": "approve", "publish_challenge": "publish", "retire_challenge": "retire"}[name]
        return _api("POST", f"/api/v1/challenges/{_require_argument(arguments, 'challenge_id')}:{suffix}", {
            "evidenceSha256": _require_argument(arguments, "evidence_sha256"),
        })
    if name == "render_challenge":
        _require_mutations()
        challenge_id = _require_argument(arguments, "challenge_id")
        return _api("POST", f"/api/v1/challenges/{challenge_id}:render", arguments.get("request") or {})
    if name == "create_render_batch":
        _require_mutations()
        _require_trusted()
        return _api(
            "POST",
            "/api/v2/render-batches",
            {"items": _require_argument(arguments, "items")},
            extra_headers={"idempotency-key": str(_require_argument(arguments, "idempotency_key"))},
        )
    if name == "get_render_batch":
        _require_trusted()
        return _api("GET", f"/api/v2/render-batches/{_require_argument(arguments, 'job_id')}")
    if name == "cancel_render_batch":
        _require_mutations()
        _require_trusted()
        return _api("POST", f"/api/v2/render-batches/{_require_argument(arguments, 'job_id')}:cancel", {})
    if name == "create_context_search":
        _require_mutations()
        _require_trusted()
        return _api(
            "POST",
            "/api/v2/context-searches",
            {
                "parentTraceReference": _require_argument(arguments, "parent_trace_reference"),
                "decisionId": _require_argument(arguments, "decision_id"),
                "contextProofHashes": _require_argument(arguments, "context_proof_hashes"),
            },
            extra_headers={"idempotency-key": str(_require_argument(arguments, "idempotency_key"))},
        )
    if name == "get_context_search":
        _require_trusted()
        return _api("GET", f"/api/v2/context-searches/{_require_argument(arguments, 'job_id')}")
    if name == "cancel_context_search":
        _require_mutations()
        _require_trusted()
        return _api("POST", f"/api/v2/context-searches/{_require_argument(arguments, 'job_id')}:cancel", {})
    if name == "create_natural_sakka_generation":
        _require_mutations()
        _require_trusted()
        return _api(
            "POST",
            "/api/v2/sakka-generations",
            {
                "rootSeed": int(_require_argument(arguments, "root_seed")),
                "initialDealer": str(arguments.get("initial_dealer", "east")),
            },
            extra_headers={
                "idempotency-key": str(_require_argument(arguments, "idempotency_key"))
            },
        )
    if name == "get_natural_sakka_generation":
        _require_trusted()
        return _api(
            "GET", f"/api/v2/sakka-generations/{_require_argument(arguments, 'job_id')}"
        )
    if name == "cancel_natural_sakka_generation":
        _require_mutations()
        _require_trusted()
        return _api(
            "POST",
            f"/api/v2/sakka-generations/{_require_argument(arguments, 'job_id')}:cancel",
            {},
        )
    if name == "list_foundry_sakkas":
        _require_trusted()
        query = {"limit": str(int(arguments.get("limit", 100)))}
        if arguments.get("provenance"):
            query["provenance"] = str(arguments["provenance"])
        if arguments.get("after"):
            query["after"] = str(arguments["after"])
        return _api("GET", f"/api/v2/sakkas?{urlencode(query)}")
    if name == "create_natural_sakka":
        _require_mutations()
        _require_trusted()
        return _api("POST", "/api/v2/sakkas/natural", {
            "traceReferences": _require_argument(arguments, "trace_references"),
            "target": int(arguments.get("target", 152)),
        })
    if name == "create_curated_sakka":
        _require_mutations()
        _require_trusted()
        return _api("POST", "/api/v2/sakkas/curated", {
            "selections": _require_argument(arguments, "selections"),
            "target": int(arguments.get("target", 152)),
        })
    if name == "get_foundry_sakka":
        _require_trusted()
        return _api("GET", f"/api/v2/sakkas/{_require_argument(arguments, 'sakka_hash')}")
    if name == "audit_game":
        identifier = _require_argument(arguments, "identifier")
        return _api("POST", f"/api/v1/history/games/{identifier}:audit", {
            "mode": "observer_safe", "rerun": bool(arguments.get("rerun", False)),
            "maximumDecisions": int(arguments.get("maximum_decisions", 12)),
        })
    if name == "explain_decision":
        decision_id = _require_argument(arguments, "decision_id")
        return _api("POST", f"/api/v1/history/decisions/{decision_id}:explain", {
            "rerun": bool(arguments.get("rerun", False)), "samples": int(arguments.get("samples", 96)),
        })
    if name == "compare_decision_versions":
        return _api("GET", f"/api/v1/history/decisions/{_require_argument(arguments, 'decision_id')}/comparison")
    if name == "record_challenge_attempt":
        _require_mutations()
        challenge_id = _require_argument(arguments, "challenge_id")
        return _api("POST", f"/api/v1/challenges/{challenge_id}/attempts", {
            "playerId": arguments.get("player_id"), "chosenAction": _require_argument(arguments, "chosen_action"),
            "responseSeconds": arguments.get("response_seconds"), "confidence": arguments.get("confidence"),
        })
    if name == "get_challenge_difficulty":
        return _api("GET", f"/api/v1/challenges/{_require_argument(arguments, 'challenge_id')}/difficulty")
    if name == "record_decision_attempt":
        _require_mutations()
        decision_id = _require_argument(arguments, "decision_id")
        return _api("POST", f"/api/v1/history/decisions/{decision_id}/attempts", {
            "playerId": arguments.get("player_id"), "chosenAction": _require_argument(arguments, "chosen_action"),
            "responseSeconds": arguments.get("response_seconds"), "confidence": arguments.get("confidence"),
        })
    if name == "get_decision_difficulty":
        return _api("GET", f"/api/v1/history/decisions/{_require_argument(arguments, 'decision_id')}/difficulty")
    if name == "get_video_record":
        return _api("GET", f"/api/v1/videos/{_require_argument(arguments, 'video_id')}")
    if name == "get_video_timeline":
        return _api("GET", f"/api/v1/videos/{_require_argument(arguments, 'video_id')}/timeline")
    if name == "get_video_audit_map":
        return _api("GET", f"/api/v1/videos/{_require_argument(arguments, 'video_id')}/audit-map")
    if name == "render_game_replay":
        _require_mutations()
        identifier = _require_argument(arguments, "identifier")
        return _api("POST", f"/api/v1/history/games/{identifier}/video-jobs", {
            "decisionId": arguments.get("decision_id"),
            "rendererPlugin": str(arguments.get("renderer_plugin", "premium_video")),
            "theme": str(arguments.get("theme", "emerald")),
            "width": int(arguments.get("width", 1080)),
            "height": int(arguments.get("height", 1920)),
            "fps": int(arguments.get("fps", 30)),
            "crf": int(arguments.get("crf", 18)),
            "auditSamples": int(arguments.get("audit_samples", 96)),
        })
    if name == "annotate_player_intent":
        _require_mutations()
        decision_id = _require_argument(arguments, "decision_id")
        return _api("POST", f"/api/v1/history/decisions/{decision_id}/intent", {
            "preferredAction": _require_argument(arguments, "selected_action"),
            "confidence": float(arguments.get("confidence", 0.8)),
            "rationale": str(_require_argument(arguments, "rationale")),
            "source": "player",
            "reviewer": arguments.get("player_id"),
        })
    if name == "adaptive_next_challenge":
        player_id = arguments.get("player_id")
        recent = ",".join(str(value) for value in arguments.get("recent_ids", []))
        path = f"/api/v1/adaptive/challenges/next?recent={recent}"
        if player_id:
            path += f"&playerId={player_id}"
        return _api("GET", path)
    if name == "list_shared_boards":
        return _api("GET", f"/api/v1/shared/boards?limit={int(arguments.get('limit', 100))}")
    if name == "verify_shared_board":
        return _api("GET", f"/api/v1/shared/boards/{_require_argument(arguments, 'identifier')}/verification")
    if name == "create_shared_board":
        _require_mutations()
        return _api("POST", "/api/v1/shared/boards", _require_argument(arguments, "request"))
    if name == "create_shared_room":
        _require_mutations()
        return _api("POST", "/api/v1/shared/rooms", {
            "boardIdentifier": _require_argument(arguments, "board_identifier"),
            "mode": str(arguments.get("mode", "async")),
            "maxPlayers": int(arguments.get("max_players", 16)),
            "allowGuests": bool(arguments.get("allow_guests", True)),
            "settings": dict(arguments.get("settings") or {}),
        })
    if name == "get_shared_room":
        return _api("GET", f"/api/v1/shared/rooms/{_require_argument(arguments, 'identifier')}")
    if name == "join_shared_room":
        _require_mutations()
        return _api("POST", f"/api/v1/shared/rooms/{_require_argument(arguments, 'identifier')}:join", {})
    if name == "start_shared_run":
        _require_mutations()
        return _api("POST", f"/api/v1/shared/rooms/{_require_argument(arguments, 'identifier')}/runs", {})
    if name == "reopen_shared_run":
        return _api("GET", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}")
    if name == "verify_shared_run":
        return _api("GET", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/verification")
    if name == "play_shared_action":
        _require_mutations()
        return _api("POST", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/actions", _require_argument(arguments, "action"))
    if name == "get_shared_run_review":
        value = "true" if bool(arguments.get("ensure_audit", False)) else "false"
        return _api("GET", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/review?ensureAudit={value}")
    if name == "explain_shared_decision":
        query = urlencode({
            "level": str(arguments.get("level", "basic")),
            "surface": str(arguments.get("surface", "replay")),
            "challengeRevealed": str(bool(arguments.get("challenge_revealed", False))).lower(),
            "ensureAudit": str(bool(arguments.get("ensure_audit", False))).lower(),
            "viewMode": str(arguments.get("view_mode", "observer")),
        })
        return _api("GET", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/decisions/{_require_argument(arguments, 'decision_id')}/explanation?{query}")
    if name == "get_shared_decision_reviews":
        return _api("GET", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/decisions/{_require_argument(arguments, 'decision_id')}/reviews")
    if name == "review_shared_decision":
        _require_mutations()
        return _api("POST", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/decisions/{_require_argument(arguments, 'decision_id')}/reviews", {
            "preferredAction": _require_argument(arguments, "preferred_action"),
            "confidence": float(arguments.get("confidence", 0.8)),
            "reasonCodes": list(arguments.get("reason_codes") or ["other"]),
            "rationale": str(_require_argument(arguments, "rationale")),
            "evidence": list(arguments.get("evidence") or []),
            "reviewerRole": "player",
            "viewMode": str(arguments.get("view_mode", "observer")),
        })
    if name == "share_shared_decision":
        _require_mutations()
        return _api("POST", f"/api/v1/shared/runs/{_require_argument(arguments, 'run_id')}/actions/{int(_require_argument(arguments, 'sequence'))}:share", {
            "scope": str(arguments.get("scope", "laffa")),
            "maximumRounds": int(arguments.get("maximum_rounds", 16)),
            "label": arguments.get("label"),
            "visibility": "unlisted",
        })
    if name == "get_shared_leaderboard":
        return _api("GET", f"/api/v1/shared/rooms/{_require_argument(arguments, 'identifier')}/leaderboard")
    if name == "compare_shared_runs":
        return _api("POST", "/api/v1/shared/comparisons", {
            "runA": _require_argument(arguments, "run_a"),
            "runB": _require_argument(arguments, "run_b"),
        })
    if name == "list_sakka_boards":
        return _api("GET", f"/api/v1/sakka/boards?limit={int(arguments.get('limit', 100))}")
    if name == "verify_sakka_board":
        return _api("GET", f"/api/v1/sakka/boards/{_require_argument(arguments, 'identifier')}/verification")
    if name == "create_sakka_board":
        _require_mutations()
        return _api("POST", "/api/v1/sakka/boards", _require_argument(arguments, "request"))
    if name == "create_sakka_room":
        _require_mutations()
        return _api("POST", "/api/v1/sakka/rooms", {
            "boardIdentifier": _require_argument(arguments, "board_identifier"),
            "mode": str(arguments.get("mode", "async")),
            "maxPlayers": int(arguments.get("max_players", 16)),
            "allowGuests": bool(arguments.get("allow_guests", True)),
        })
    if name == "get_sakka_room":
        return _api("GET", f"/api/v1/sakka/rooms/{_require_argument(arguments, 'identifier')}")
    if name == "join_sakka_room":
        _require_mutations()
        return _api("POST", f"/api/v1/sakka/rooms/{_require_argument(arguments, 'identifier')}:join", {})
    if name == "start_sakka_run":
        _require_mutations()
        return _api("POST", f"/api/v1/sakka/rooms/{_require_argument(arguments, 'identifier')}/runs", {})
    if name == "play_sakka_action":
        _require_mutations()
        return _api("POST", f"/api/v1/sakka/runs/{_require_argument(arguments, 'run_id')}/actions", _require_argument(arguments, "action"))
    if name == "reopen_sakka_run":
        return _api("GET", f"/api/v1/sakka/runs/{_require_argument(arguments, 'run_id')}")
    if name == "verify_sakka_run":
        return _api("GET", f"/api/v1/sakka/runs/{_require_argument(arguments, 'run_id')}/verification")
    if name == "get_sakka_leaderboard":
        return _api("GET", f"/api/v1/sakka/rooms/{_require_argument(arguments, 'identifier')}/leaderboard")
    if name == "compare_sakka_runs":
        return _api("POST", "/api/v1/sakka/comparisons", {
            "runA": _require_argument(arguments, "run_a"),
            "runB": _require_argument(arguments, "run_b"),
        })
    if name == "audit_sakka_run":
        _require_mutations()
        return _api("POST", f"/api/v1/sakka/runs/{_require_argument(arguments, 'run_id')}:audit", {
            "deep": bool(arguments.get("deep", False)), "force": bool(arguments.get("force", False)),
            "maximumDecisions": int(arguments.get("maximum_decisions", 32)),
        })
    if name == "share_sakka_decision":
        _require_mutations()
        return _api("POST", f"/api/v1/sakka/runs/{_require_argument(arguments, 'run_id')}/actions/{int(_require_argument(arguments, 'sequence'))}:share", {
            "scope": str(arguments.get("scope", "round")), "maximumRounds": 16,
            "label": arguments.get("label"), "visibility": "unlisted",
        })
    if name == "review_sakka_decision":
        _require_mutations()
        return _api("POST", f"/api/v1/sakka/runs/{_require_argument(arguments, 'run_id')}/actions/{int(_require_argument(arguments, 'sequence'))}/feedback", {
            "preferredAction": _require_argument(arguments, "preferred_action"),
            "confidence": float(arguments.get("confidence", 0.8)),
            "rationale": str(_require_argument(arguments, "rationale")),
            "interestingness": float(arguments.get("interestingness", 0.0)),
        })
    if name == "list_live_rooms":
        status = arguments.get("status")
        suffix = f"?limit={int(arguments.get('limit', 100))}" + (f"&status={status}" if status else "")
        return _api("GET", "/api/v1/live/rooms" + suffix)
    if name == "create_live_room":
        _require_mutations(); return _api("POST", "/api/v1/live/rooms", _require_argument(arguments, "request"))
    if name == "get_live_room":
        identifier = _require_argument(arguments, "identifier")
        return _api("GET", f"/api/v1/live/rooms/{identifier}?sinceSequence={int(arguments.get('since_sequence', 0))}")
    if name == "claim_live_seat":
        _require_mutations(); return _api("POST", f"/api/v1/live/rooms/{_require_argument(arguments, 'identifier')}/seats", {
            "seat": _require_argument(arguments, "seat"), "displayName": _require_argument(arguments, "display_name"),
        })
    if name == "set_live_ready":
        _require_mutations(); return _api("POST", f"/api/v1/live/rooms/{_require_argument(arguments, 'identifier')}/ready", {"ready": bool(arguments.get("ready", True))})
    if name == "start_live_table":
        _require_mutations(); return _api("POST", f"/api/v1/live/rooms/{_require_argument(arguments, 'identifier')}/start", {})
    if name == "play_live_action":
        _require_mutations(); return _api("POST", f"/api/v1/live/rooms/{_require_argument(arguments, 'identifier')}/actions", _require_argument(arguments, "action"))
    if name == "verify_live_room":
        return _api("GET", f"/api/v1/live/rooms/{_require_argument(arguments, 'identifier')}/verification")
    if name == "list_adjudication_cases":
        parts = [f"limit={int(arguments.get('limit', 100))}"]
        if arguments.get("game_id"): parts.append(f"gameId={arguments['game_id']}")
        if arguments.get("status"): parts.append(f"status={arguments['status']}")
        return _api("GET", "/api/v1/adjudication/cases?" + "&".join(parts))
    if name == "open_adjudication_case":
        _require_mutations(); return _api("POST", "/api/v1/adjudication/cases", _require_argument(arguments, "request"))
    if name == "get_adjudication_case":
        return _api("GET", f"/api/v1/adjudication/cases/{_require_argument(arguments, 'case_id')}")
    if name == "recommend_adjudication_ruling":
        return _api("GET", f"/api/v1/adjudication/cases/{_require_argument(arguments, 'case_id')}/recommendation")
    if name == "get_security_capabilities":
        return _api("GET", "/api/v1/security/capabilities")
    if name == "get_hitl_profile":
        return _api("GET", "/api/v1/hitl/profile")
    if name == "list_hitl_cases":
        return _api("GET", f"/api/v1/hitl/cases?limit={int(arguments.get('limit',100))}")
    if name == "create_hitl_case":
        _require_mutations(); return _api("POST", "/api/v1/hitl/cases", {"decisionId":_require_argument(arguments,"decision_id"),"correctionType":str(arguments.get("correction_type","analysis_defect")),"highImpact":bool(arguments.get("high_impact",False))})
    if name == "get_hitl_case":
        blind="true" if bool(arguments.get("blind",False)) else "false"
        return _api("GET", f"/api/v1/hitl/cases/{_require_argument(arguments,'case_id')}?blind={blind}")
    if name == "assign_hitl_review":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/cases/{_require_argument(arguments,'case_id')}/assignments", {"deviceId":arguments.get("device_id","mcp"),"networkId":arguments.get("network_id","mcp")})
    if name == "submit_hitl_review":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/cases/{_require_argument(arguments,'case_id')}/submissions", {"assignmentId":_require_argument(arguments,"assignment_id"),"assignmentToken":_require_argument(arguments,"assignment_token"),"preferredAction":_require_argument(arguments,"preferred_action"),"confidence":float(arguments.get("confidence",.8)),"reasonCodes":list(arguments.get("reason_codes") or ["other"]),"rationaleAr":_require_argument(arguments,"rationale_ar"),"claims":list(arguments.get("claims") or []),"evidence":list(arguments.get("evidence") or []),"viewMode":"observer","hiddenCardsSeen":bool(arguments.get("hidden_cards_seen",False)),"finalResultSeen":bool(arguments.get("final_result_seen",False)),"disclosureTiming":"after_reveal" if arguments.get("hidden_cards_seen") or arguments.get("final_result_seen") else "before_reveal"})
    if name == "compute_hitl_consensus":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/cases/{_require_argument(arguments,'case_id')}/consensus", {})
    if name == "set_hitl_consent":
        _require_mutations(); return _api("PUT", "/api/v1/hitl/consent", {"allowTraining":bool(arguments.get("allow_training",False)),"allowResearch":bool(arguments.get("allow_research",False)),"allowPublicChallenge":bool(arguments.get("allow_public_challenge",False)),"allowVoice":bool(arguments.get("allow_voice",False)),"allowIdentityAttribution":bool(arguments.get("allow_identity_attribution",False)),"allowPrivateCoaching":bool(arguments.get("allow_private_coaching",True)),"revoke":bool(arguments.get("revoke",False))})
    if name == "create_hitl_training_candidate":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/cases/{_require_argument(arguments,'case_id')}/training-candidates", {})
    if name == "record_hitl_challenger_evaluation":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/training-candidates/{_require_argument(arguments,'candidate_id')}/evaluations", {"validatorId":_require_argument(arguments,"validator_id"),"validatorType":arguments.get("validator_type","independent_solver"),"status":_require_argument(arguments,"status"),"metrics":arguments.get("metrics",{}),"independent":bool(arguments.get("independent",True))})
    if name == "create_engine_improvement_proposal":
        _require_mutations(); return _api("POST", "/api/v1/hitl/improvements", {"target":_require_argument(arguments,"target"),"titleAr":_require_argument(arguments,"title_ar"),"rationaleAr":_require_argument(arguments,"rationale_ar"),"sourceCaseIds":list(arguments.get("source_case_ids") or []),"requestedChange":arguments.get("requested_change",{})})
    if name == "evaluate_engine_improvement":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/improvements/{_require_argument(arguments,'proposal_id')}/evaluations", {"validatorId":_require_argument(arguments,"validator_id"),"status":_require_argument(arguments,"status"),"evidence":arguments.get("evidence",{}),"independent":bool(arguments.get("independent",True))})
    if name == "render_hitl_review_clip":
        _require_mutations(); return _api("POST", f"/api/v1/hitl/cases/{_require_argument(arguments,'case_id')}/review-clip", {"width":arguments.get("width"),"height":arguments.get("height"),"fps":arguments.get("fps")})
    raise MCPError(-32602, f"unknown tool: {name}")


def _read_resource(uri: str) -> tuple[str, str]:
    if uri == "baloot://bundle/current":
        return "application/json", json.dumps(build_bundle(ROOT).to_dict(), ensure_ascii=False)
    if uri == "baloot://rules/current":
        return "application/json", json.dumps(build_bundle(ROOT).rules.data, ensure_ascii=False)
    if uri == "baloot://agents/profiles":
        return "application/json", json.dumps(_api("GET", "/api/v1/agents/profiles"), ensure_ascii=False)
    if uri == "baloot://learning/status":
        return "application/json", json.dumps({
            "datasets": _api("GET", "/api/v1/learning/datasets"),
            "models": _api("GET", "/api/v1/learning/models"),
        }, ensure_ascii=False)
    if uri == "baloot://audit/profile":
        return "application/json", json.dumps(build_bundle(ROOT).component("audit"), ensure_ascii=False)
    if uri == "baloot://difficulty/profile":
        return "application/json", json.dumps(build_bundle(ROOT).component("difficulty"), ensure_ascii=False)
    if uri == "baloot://shared/profile":
        return "application/json", json.dumps(build_bundle(ROOT).component("shared"), ensure_ascii=False)
    if uri == "baloot://sakka/profile":
        return "application/json", json.dumps(build_bundle(ROOT).component("duplicate"), ensure_ascii=False)
    if uri == "baloot://live/profile":
        return "application/json", json.dumps(build_bundle(ROOT).component("live"), ensure_ascii=False)
    if uri == "baloot://history/canonical":
        return "application/json", json.dumps({"schemaVersion": 17, "eventStore": "canonical", "features": ["append_only_events", "encrypted_snapshots", "branches", "corrections", "verification"]}, ensure_ascii=False)
    if uri == "baloot://adjudication/profile":
        path = ROOT / "configs/adjudication/saudi_researched_v1.yaml"
        import yaml
        return "application/json", json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")), ensure_ascii=False)
    if uri == "baloot://security/capabilities":
        return "application/json", json.dumps(_api("GET", "/api/v1/security/capabilities"), ensure_ascii=False)
    if uri == "baloot://hitl/profile":
        return "application/json", json.dumps(build_bundle(ROOT).component("hitl"), ensure_ascii=False)
    if uri == "baloot://configuration/registry":
        return "application/json", json.dumps(_api("GET", "/api/v1/configuration-registry/profile"), ensure_ascii=False)
    if uri == "baloot://plugins/policy":
        return "application/json", json.dumps(_api("GET", "/api/v1/plugins"), ensure_ascii=False)
    if uri == "baloot://performance/profile":
        return "application/json", json.dumps(_api("GET", "/api/v1/performance/profile"), ensure_ascii=False)
    raise MCPError(-32002, f"resource not found: {uri}")


class MCPTaskStore:
    """Small durable task state store for the MCP 2025-11-25 experimental task contract."""

    def __init__(self, path: Path):
        self.path = path; path.parent.mkdir(parents=True, exist_ok=True); self._lock = threading.RLock()
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS mcp_tasks(task_id TEXT PRIMARY KEY,status TEXT NOT NULL,request_json TEXT NOT NULL,result_json TEXT,error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)")

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        from datetime import datetime, timezone
        task_id = "MTASK-" + secrets.token_urlsafe(14); now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO mcp_tasks VALUES(?,?,?,?,?,?,?)", (task_id,"working",json.dumps(request,ensure_ascii=False),None,None,now,now))
        return self.get(task_id)

    def finish(self, task_id: str, *, result: Any | None = None, error: str | None = None) -> dict[str, Any]:
        from datetime import datetime, timezone
        status = "failed" if error else "completed"; now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE mcp_tasks SET status=?,result_json=?,error=?,updated_at=? WHERE task_id=?",
                         (status, json.dumps(result,ensure_ascii=False) if result is not None else None, error, now, task_id))
        return self.get(task_id)

    def cancel(self, task_id: str) -> dict[str, Any]:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE mcp_tasks SET status='cancelled',updated_at=? WHERE task_id=? AND status='working'", (now,task_id))
        return self.get(task_id)

    def get(self, task_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.path) as conn:
            row=conn.execute("SELECT task_id,status,request_json,result_json,error,created_at,updated_at FROM mcp_tasks WHERE task_id=?",(task_id,)).fetchone()
        if row is None: raise KeyError(task_id)
        return {"taskId":row[0],"status":row[1],"request":json.loads(row[2]),"result":json.loads(row[3]) if row[3] else None,"error":row[4],"createdAt":row[5],"updatedAt":row[6]}

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            rows=conn.execute("SELECT task_id FROM mcp_tasks ORDER BY created_at DESC LIMIT ?",(max(1,min(500,limit)),)).fetchall()
        return [self.get(str(row[0])) for row in rows]


_MCP_TASKS = MCPTaskStore(data_root() / "mcp" / "tasks.sqlite3")


@dataclass(slots=True)
class MCPProtocol:
    initialized: bool = False
    negotiated_version: str = LATEST_PROTOCOL_VERSION

    def process(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            raise MCPError(-32600, "invalid JSON-RPC request")
        method = message["method"]
        request_id = message.get("id")
        is_notification = "id" not in message
        params = message.get("params") or {}
        if not isinstance(params, dict):
            raise MCPError(-32602, "params must be an object")

        if method == "initialize":
            client_version = str(params.get("protocolVersion", ""))
            self.negotiated_version = client_version if client_version in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            self.initialized = True
            result = {
                "protocolVersion": self.negotiated_version,
                "capabilities": {
                    "resources": {"subscribe": False, "listChanged": False},
                    "tools": {"listChanged": False},
                    "tasks": {"list": {}, "cancel": {}, "requests": {"tools": {"call": {}}}},
                },
                "serverInfo": {
                    "name": "baloot-platform",
                    "title": "Baloot Platform",
                    "version": SERVER_VERSION,
                    "description": "Authoritative configurable Baloot engine, Research V4, games, and governed rendering",
                },
                "instructions": (
                    "Read-only operations are available by default. Mutations and trusted all-hands contexts "
                    "require separate server-side opt-in."
                ),
            }
            return None if is_notification else self._result(request_id, result)

        if method in {"notifications/initialized", "notifications/cancelled"}:
            return None
        if not self.initialized and method != "ping":
            raise MCPError(-32002, "server has not been initialized")
        if method == "ping":
            return None if is_notification else self._result(request_id, {})
        if method == "tasks/list":
            return None if is_notification else self._result(request_id, {"tasks": _MCP_TASKS.list(int(params.get("limit", 100)))})
        if method == "tasks/get":
            try:
                task = _MCP_TASKS.get(str(params.get("taskId") or params.get("task_id")))
            except KeyError as exc:
                raise MCPError(-32002, "task not found") from exc
            return None if is_notification else self._result(request_id, {"task": task})
        if method == "tasks/cancel":
            try:
                task = _MCP_TASKS.cancel(str(params.get("taskId") or params.get("task_id")))
            except KeyError as exc:
                raise MCPError(-32002, "task not found") from exc
            return None if is_notification else self._result(request_id, {"task": task})
        if method == "tools/list":
            result = {"tools": list(TOOL_DEFINITIONS)}
            return None if is_notification else self._result(request_id, result)
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise MCPError(-32602, "tools/call requires name and object arguments")
            task_requested = bool(params.get("task") or (params.get("_meta") or {}).get("task"))
            task_record = _MCP_TASKS.create({"tool": name, "arguments": arguments}) if task_requested else None
            try:
                value = _call_tool(name, arguments)
                if task_record is not None:
                    task_record = _MCP_TASKS.finish(task_record["taskId"], result=value)
                result = {
                    "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                    "structuredContent": value,
                    "isError": False,
                }
                if task_record is not None:
                    result["task"] = task_record
            except MCPError:
                raise
            except Exception as exc:  # tool failures are ordinary MCP tool errors
                if task_record is not None:
                    task_record = _MCP_TASKS.finish(task_record["taskId"], error=str(exc))
                result = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
                if isinstance(exc, SearchTimeBudgetExceeded):
                    result["structuredContent"] = exc.to_detail()
                    result["content"][0]["text"] = json.dumps(exc.to_detail(), ensure_ascii=False)
                if task_record is not None:
                    result["task"] = task_record
            return None if is_notification else self._result(request_id, result)
        if method == "resources/list":
            result = {
                "resources": [
                    {
                        "uri": "baloot://bundle/current",
                        "name": "Current configuration bundle",
                        "description": "Immutable aggregate configuration bundle",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://rules/current",
                        "name": "Current Baloot rules",
                        "description": "Active deterministic rules profile",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://agents/profiles",
                        "name": "Agent profiles",
                        "description": "Configurable player intelligence profiles",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://learning/status",
                        "name": "Learning status",
                        "description": "Registered datasets and models",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://audit/profile",
                        "name": "Replay audit profile",
                        "description": "Active observer-safe replay-intelligence settings",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://difficulty/profile",
                        "name": "Difficulty calibration profile",
                        "description": "Active audience-calibration settings",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://shared/profile",
                        "name": "Standardized shared لفة / صكّة profile",
                        "description": "Frozen same-start environment, agent tuning and reveal policy",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://sakka/profile",
                        "name": "Sakka same-start profile",
                        "description": "Active shareable same-start round and match configuration",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://live/profile",
                        "name": "Synchronous live-table profile",
                        "description": "Four-human table, reconnect, timer and referee settings",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://adjudication/profile",
                        "name": "Adjudication profile",
                        "description": "Versioned violation, ruling, appeal and correction policy",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://security/capabilities",
                        "name": "Security capabilities",
                        "description": "Optional identity, roles, scopes, OIDC and passkey boundaries",
                        "mimeType": "application/json",
                    },
                    {"uri": "baloot://configuration/registry", "name": "Versioned configuration registry", "description": "Published row-based configuration and governance profile", "mimeType": "application/json"},
                    {"uri": "baloot://plugins/policy", "name": "Plugin host policy", "description": "Plugin isolation, signature, permission and resource policy", "mimeType": "application/json"},
                    {"uri": "baloot://performance/profile", "name": "Performance profile", "description": "Latency budgets and observability policy", "mimeType": "application/json"},
                    {
                        "uri": "baloot://hitl/profile",
                        "name": "Governed Human-in-the-Loop profile",
                        "description": "Blind review, consent, consensus, challenger and no-auto-promotion policy",
                        "mimeType": "application/json",
                    },
                    {
                        "uri": "baloot://history/canonical",
                        "name": "Canonical history contract",
                        "description": "Append-only events, snapshots, branches and correction lineage",
                        "mimeType": "application/json",
                    },
                ]
            }
            return None if is_notification else self._result(request_id, result)
        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise MCPError(-32602, "resources/read requires uri")
            mime_type, text = _read_resource(uri)
            return None if is_notification else self._result(
                request_id,
                {"contents": [{"uri": uri, "mimeType": mime_type, "text": text}]},
            )
        if method in {"resources/templates/list", "prompts/list"}:
            key = "resourceTemplates" if method.startswith("resources/") else "prompts"
            return None if is_notification else self._result(request_id, {key: []})
        raise MCPError(-32601, f"method not found: {method}")

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def error_response(request_id: Any, error: MCPError) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": error.code, "message": error.message}
        if error.data is not None:
            payload["data"] = error.data
        return {"jsonrpc": "2.0", "id": request_id, "error": payload}


def run_stdio() -> None:
    protocol = MCPProtocol()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            message = json.loads(line)
            if isinstance(message, dict):
                request_id = message.get("id")
            response = protocol.process(message)
        except json.JSONDecodeError as exc:
            response = MCPProtocol.error_response(None, MCPError(-32700, "parse error", str(exc)))
        except MCPError as exc:
            response = MCPProtocol.error_response(request_id, exc)
        except SearchTimeBudgetExceeded as exc:
            response = MCPProtocol.error_response(
                request_id, MCPError(-32000, "time-censored evidence", exc.to_detail())
            )
        except Exception as exc:  # defensive protocol boundary
            response = MCPProtocol.error_response(request_id, MCPError(-32603, "internal error", str(exc)))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def _allowed_origin(origin: str | None) -> bool:
    if not origin:
        return True
    configured = os.environ.get("BALOOT_MCP_ALLOWED_ORIGINS", "http://127.0.0.1,http://localhost")
    allowed = {item.strip() for item in configured.split(",") if item.strip()}
    return "*" in allowed or origin in allowed


def _authorized_scopes(authorization: str | None) -> frozenset[str] | None:
    environment = os.environ.get("BALOOT_ENV", "development").lower()
    expected = os.environ.get("BALOOT_MCP_AUTH_TOKEN", "")
    static_allowed = environment in {"development", "test"} or _truthy("BALOOT_MCP_ALLOW_STATIC_TOKEN")
    if expected and static_allowed and authorization == f"Bearer {expected}":
        return frozenset(Scope(item).value for item in Scope)
    secret = os.environ.get("BALOOT_ACCESS_TOKEN_SECRET", "").encode("utf-8")
    if authorization and authorization.lower().startswith("bearer ") and len(secret) >= 32:
        token = authorization[7:].strip()
        audience = os.environ.get("BALOOT_MCP_TOKEN_AUDIENCE", "baloot-mcp")
        try:
            claims = SignedAccessTokenService(
                secret, issuer=os.environ.get("BALOOT_TOKEN_ISSUER", "baloot-platform"), audience=audience,
            ).verify(token, required_audience=audience)
        except ValueError:
            return None
        scopes = frozenset(claims.scopes)
        return scopes if Scope.READ.value in scopes else None
    if not expected and not secret:
        return frozenset(Scope(item).value for item in Scope) if environment in {"development", "test"} else None
    return None


def _authorization_context(authorization: str | None) -> dict[str, Any] | None:
    scopes = _authorized_scopes(authorization)
    if scopes is None: return None
    subject = os.environ.get("BALOOT_MCP_SUBJECT_ID", "mcp-local")
    tenant = os.environ.get("BALOOT_MCP_TENANT_ID", "")
    secret = os.environ.get("BALOOT_ACCESS_TOKEN_SECRET", "").encode("utf-8")
    if authorization and authorization.lower().startswith("bearer ") and len(secret) >= 32:
        try:
            audience = os.environ.get("BALOOT_MCP_TOKEN_AUDIENCE", "baloot-mcp")
            claims = SignedAccessTokenService(secret, issuer=os.environ.get("BALOOT_TOKEN_ISSUER", "baloot-platform"), audience=audience).verify(authorization[7:].strip(), required_audience=audience)
            subject = claims.subject_id; tenant = claims.tenant_id or ""
        except ValueError:
            return None
    return {"scopes": scopes, "subject": subject, "tenant": tenant, "authorization": authorization}


def _protected_resource_metadata(request: Request, servers: list[str]) -> dict[str, Any]:
    configured = os.environ.get("BALOOT_MCP_RESOURCE_URL", "").strip()
    resource = configured or str(request.base_url).rstrip("/") + "/mcp"
    return {
        "resource": resource,
        "authorization_servers": servers,
        "scopes_supported": [scope.value for scope in Scope],
        "bearer_methods_supported": ["header"],
        "resource_documentation": str(request.base_url).rstrip("/") + "/healthz",
    }


def create_http_app() -> FastAPI:
    servers = [
        item.strip()
        for item in os.environ.get("BALOOT_MCP_AUTHORIZATION_SERVERS", "").split(",")
        if item.strip()
    ]
    if not servers:
        servers = sorted({provider.issuer for provider in load_oidc_providers().values()})
    app = FastAPI(title="Baloot Platform MCP", version=SERVER_VERSION, docs_url=None, redoc_url=None)
    sessions: dict[str, float] = {}

    @app.get("/healthz")
    def health() -> dict[str, str]: return {"status": "ok", "version": SERVER_VERSION}

    @app.get("/.well-known/oauth-protected-resource")
    @app.get("/.well-known/oauth-protected-resource/mcp")
    def protected_resource_metadata(request: Request) -> dict[str, Any]:
        return _protected_resource_metadata(request, servers)

    @app.get("/mcp")
    async def mcp_get(request: Request, origin: str | None = Header(default=None),
                      authorization: str | None = Header(default=None),
                      mcp_session_id: str | None = Header(default=None, alias="MCP-Session-Id")) -> Response:
        if not _allowed_origin(origin): return JSONResponse(status_code=403, content={"error": "origin not allowed"})
        context = _authorization_context(authorization)
        if context is None:
            metadata_url = str(request.base_url).rstrip("/") + "/.well-known/oauth-protected-resource/mcp"
            return JSONResponse(status_code=401, content={"error": "unauthorized"}, headers={"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'})
        session_id = mcp_session_id or secrets.token_urlsafe(18); sessions[session_id] = __import__("time").time()
        async def stream():
            endpoint = json.dumps({"sessionId": session_id, "protocolVersion": LATEST_PROTOCOL_VERSION}, separators=(",", ":"))
            yield f"event: endpoint\ndata: {endpoint}\n\n"
            await asyncio.sleep(0)
            yield ": keepalive\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"MCP-Session-Id": session_id, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/mcp")
    async def mcp_post(request: Request, origin: str | None = Header(default=None),
                       authorization: str | None = Header(default=None),
                       mcp_protocol_version: str | None = Header(default=None, alias="MCP-Protocol-Version"),
                       mcp_session_id: str | None = Header(default=None, alias="MCP-Session-Id")) -> Response:
        if not _allowed_origin(origin): return JSONResponse(status_code=403, content={"error": "origin not allowed"})
        context = _authorization_context(authorization)
        if context is None:
            metadata_url = str(request.base_url).rstrip("/") + "/.well-known/oauth-protected-resource/mcp"
            return JSONResponse(status_code=401, content={"error": "unauthorized"}, headers={"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'})
        if mcp_protocol_version and mcp_protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            return JSONResponse(status_code=400, content={"error": "unsupported MCP protocol version"})
        try: message = await request.json()
        except Exception as exc: return JSONResponse(status_code=400, content=MCPProtocol.error_response(None, MCPError(-32700, "parse error", str(exc))))
        session_id = mcp_session_id or secrets.token_urlsafe(18); sessions[session_id] = __import__("time").time()
        tokens = (
            _REQUEST_SCOPES.set(context["scopes"]), _REQUEST_AUTHORIZATION.set(context["authorization"]),
            _REQUEST_SUBJECT.set(context["subject"]), _REQUEST_TENANT.set(context["tenant"]),
        )
        try:
            protocol = MCPProtocol(initialized=True, negotiated_version=mcp_protocol_version or LATEST_PROTOCOL_VERSION)
            if isinstance(message, dict) and message.get("method") == "initialize": protocol.initialized = False
            request_id = message.get("id") if isinstance(message, dict) else None
            try: response = protocol.process(message)
            except MCPError as exc: response = MCPProtocol.error_response(request_id, exc)
            except SearchTimeBudgetExceeded as exc:
                response = MCPProtocol.error_response(
                    request_id, MCPError(-32000, "time-censored evidence", exc.to_detail())
                )
            except Exception as exc: response = MCPProtocol.error_response(request_id, MCPError(-32603, "internal error", str(exc)))
            if response is None: return Response(status_code=202, headers={"MCP-Session-Id": session_id})
            return JSONResponse(status_code=200, content=response, media_type="application/json", headers={"MCP-Session-Id": session_id})
        finally:
            _REQUEST_SCOPES.reset(tokens[0]); _REQUEST_AUTHORIZATION.reset(tokens[1]); _REQUEST_SUBJECT.reset(tokens[2]); _REQUEST_TENANT.reset(tokens[3])

    return app


def main() -> None:
    transport = os.environ.get("BALOOT_MCP_TRANSPORT", "stdio").lower()
    if transport == "stdio":
        run_stdio()
        return
    if transport in {"streamable-http", "http"}:
        import uvicorn

        bind = os.environ.get("BALOOT_MCP_BIND", "127.0.0.1:8000")
        host, port_text = bind.rsplit(":", 1)
        uvicorn.run(create_http_app(), host=host, port=int(port_text), log_level=os.environ.get("BALOOT_LOG_LEVEL", "info"))
        return
    raise SystemExit(f"unsupported MCP transport: {transport}")


if __name__ == "__main__":
    main()
