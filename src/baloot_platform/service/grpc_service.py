"""Typed gRPC adapter for the authoritative Baloot application services.

Generated protobuf modules are build artifacts. Generate them with
``python scripts/generate_grpc.py`` after installing ``baloot-platform[grpc]``.
No Baloot rules are duplicated in this adapter.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import hmac
import json
import os
import uuid
from pathlib import Path
from typing import Any

from baloot_platform.audit import build_video_audit_map
from baloot_platform.analysis.search_errors import SearchTimeBudgetExceeded
from baloot_platform.config import data_root, platform_root
from baloot_platform.configuration.schedule import ConfigurationIntegrityError, ConfigurationScheduleConflict
from baloot_platform.identity.ownership import SERVICE_SUBJECT_OWNER, owns_subject
from baloot_platform.observability import configure_event_logging
from baloot_platform.security import Scope
from baloot_platform.service.authorization import (
    DEFAULT_TENANT, RPC_SERVICE_KEY_METHODS, Credentials, claims_identity, credentials,
    history_without_media, rpc_scope, tenant_job_rpc,
)
from baloot_platform.service.client_identity import (
    initialize_service_subject_ownership,
    service_client_identity,
    service_subject_ownership,
)
from baloot_platform.version import __version__
from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY, DatabaseIdentityContext

ROOT = platform_root()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_render_batch_request_payload(raw: bytes) -> tuple[dict[str, Any], ...]:
    """Validate the complete gRPC JSON envelope through the HTTP contract."""

    from baloot_platform.service.api import FoundryRenderBatchCreateRequest

    value = json.loads(raw) if raw else {}
    if not isinstance(value, dict):
        raise ValueError("canonical_json must contain an object")
    validated = FoundryRenderBatchCreateRequest.model_validate(value)
    return tuple(item.model_dump(exclude_none=True) for item in validated.items)


async def serve() -> None:  # pragma: no cover - optional runtime
    try:
        import grpc
        from baloot_platform.generated import (
            configuration_pb2, configuration_pb2_grpc, hitl_pb2, hitl_pb2_grpc, operations_pb2, operations_pb2_grpc, platform_pb2, platform_pb2_grpc,
            sakka_pb2, sakka_pb2_grpc, shared_pb2, shared_pb2_grpc,
        )
    except ImportError as exc:
        raise RuntimeError('Generate stubs and install with: pip install "baloot-platform[grpc]"') from exc

    from baloot_platform.analysis.evaluator import analyze
    from baloot_platform.analysis.models import InformationState
    from baloot_platform.config import build_bundle
    from baloot_platform.core.cards import Card
    from baloot_platform.core.enums import Seat, Suit
    from baloot_platform.core.serialization import continue_trusted_context, replay_trusted_context
    from baloot_platform.service.api import (
        ACCESS_TOKENS, ADJUDICATION, AUDIT, AUTHORIZATION_POLICY, CONFIGURATION, CONFIGURATION_PROFILE, DIFFICULTY, EXECUTOR, EXPLANATIONS, IDENTITY,
        HITL, HITL_PROFILE, HITL_REVIEW_RENDERER, LEARNING, LEARNING_STORE, LIVE, OUTPUTS, PERFORMANCE, PERFORMANCE_PROFILE, CAPABILITY_MATRIX, CERTIFICATION_PROFILE, BENCHMARK_PROFILE, OPERATIONS_PROFILE, REGISTRY, RENDER_BATCHES, RESEARCH, REVIEWS, SAKKA, STORE,
        HistoricalReplayRenderRequest, LegacyVideoCompatibilityError, VideoJobRequest,
        _accepted_actions_for_challenge, _accepted_actions_for_decision,
        _execute_history_replay_render_job, _legacy_video_run_is_v4,
        _legacy_video_run_view,
        _submit_legacy_video_compatibility,
        require_verified_configuration,
    )
    from baloot_platform.service.game import GameManager
    from baloot_platform.shared import (
        SharedDecisionAccessError,
        SharedGameNotFound,
        observer_relation,
        private_decision_review_allowed,
        resolve_shared_decision_access,
        resolve_shared_game_access,
    )
    from baloot_platform.benchmark import IntelligenceBenchmarkV18
    from baloot_platform.certification import verify_certification_candidate
    from baloot_platform.storage.game_store import SQLiteGameStore

    require_verified_configuration()
    initialize_service_subject_ownership()
    runtime = Path(os.environ.get("BALOOT_RUNTIME", data_root() / "runtime")).expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    manager = GameManager(ROOT, SQLiteGameStore(runtime / "grpc-games.sqlite3"))

    def _metadata(context) -> dict[str, str]:
        return {str(item.key).lower(): str(item.value) for item in context.invocation_metadata()}

    authenticated_rpc: ContextVar[Credentials] = ContextVar("authenticated_rpc")

    async def _identity(context):
        metadata = _metadata(context)
        auth = authenticated_rpc.get()
        if auth.identity is not None:
            return auth.identity
        if auth.claims is not None:
            return claims_identity(
                auth.claims, metadata.get("x-baloot-subject-name", auth.claims.subject_id),
            )
        return service_client_identity(metadata, "gRPC Client")

    async def _require_private_decision(context, decision_id: str) -> dict[str, Any]:
        decision = LEARNING_STORE.get_decision(decision_id)
        if decision is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
        try:
            access = resolve_shared_decision_access(SAKKA.store, decision)
        except SharedDecisionAccessError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        if access is not None and not access.completed:
            identity = await _identity(context)
            if not access.can_view_private(identity.subject_id):
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "private decision analysis is unavailable during an active shared run",
                )
        return decision

    async def _require_private_game(context, identifier: str, subject_id: str | None = None) -> dict[str, Any]:
        game = LEARNING_STORE.get_game(identifier)
        if game is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "archived game not found")
        try:
            access = resolve_shared_game_access(
                SAKKA.store, str(game.get("gameId") or ""), subject_id=subject_id,
            )
        except SharedGameNotFound:
            await context.abort(grpc.StatusCode.NOT_FOUND, "archived game not found")
        except SharedDecisionAccessError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        if access is not None and not access.completed:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED,
                                "aggregate game analysis is unavailable during an active shared run")
        for item in game.get("decisions") or []:
            decision = LEARNING_STORE.get_decision(str(item.get("decisionId") or ""))
            if decision is None:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "aggregate decision identity is unavailable")
            try:
                access = resolve_shared_decision_access(SAKKA.store, decision)
            except SharedDecisionAccessError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            if access is not None and not access.completed:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED,
                                    "aggregate game analysis is unavailable during an active shared run")
        return game

    def _payload(raw: bytes) -> dict[str, Any]:
        if not raw:
            return {}
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("canonical_json must encode an object")
        return value

    async def _require_trusted(context) -> None:
        auth = authenticated_rpc.get()
        if Scope.HIDDEN_READ not in AUTHORIZATION_POLICY.effective_scopes(
            role=auth.role, claims=auth.claims,
        ):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "trusted hidden-information access is not enabled")
        configured = os.environ.get("BALOOT_TRUSTED_API_KEY", "")
        supplied = _metadata(context).get("x-baloot-trusted-key", "")
        if configured:
            if not hmac.compare_digest(configured.encode(), supplied.encode()):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "trusted hidden-information access is not enabled")
            return
        if os.environ.get("BALOOT_ALLOW_TRUSTED_CONTEXT", "false").lower() in {"1", "true", "yes"}:
            return
        if (
            os.environ.get("BALOOT_ENV", "development").lower() in {"development", "test"}
            and context.peer().startswith(("ipv4:127.0.0.1:", "ipv6:[::1]:"))
        ):
            return
        await context.abort(grpc.StatusCode.PERMISSION_DENIED, "trusted hidden-information access is not enabled")

    def bundle_ref(bundle):
        components = {}
        for family, value in bundle.to_dict()["components"].items():
            components[family] = platform_pb2.ProfileRef(
                id=str(value.get("id", family)),
                version=str(value.get("version", "")),
                sha256=str(value.get("sha256", "")),
            )
        return platform_pb2.BundleRef(sha256=bundle.sha256, components=components)

    def game_state(game):
        bundle = build_bundle(ROOT, mode=game.mode)
        payload = game.to_dict(reveal_hands=False)
        return platform_pb2.GameState(
            game=platform_pb2.GameRef(game_id=game.id, version=game.version),
            bundle=bundle_ref(bundle),
            phase=game.state.phase.value,
            current_player=game.state.current_player.name.lower() if game.state.current_player is not None else "",
            canonical_json=_json_bytes(payload),
            state_sha256=__import__("hashlib").sha256(_json_bytes(payload)).hexdigest(),
        )

    class Service(platform_pb2_grpc.BalootPlatformServicer):
        async def HealthCheck(self, request, context):
            bundle = build_bundle(ROOT)
            return platform_pb2.Health(status="ok", version=__import__("baloot_platform.version", fromlist=["__version__"]).__version__, bundle_sha256=bundle.sha256)

        async def CreateGame(self, request, context):
            try:
                game = manager.create(
                    seed=request.seed,
                    dealer=Seat.parse(request.dealer or "east"),
                    mode=request.rules_profile or "full",
                    match_score=(request.us_score, request.them_score),
                )
                return game_state(game)
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetGame(self, request, context):
            try:
                return game_state(manager.get(request.game_id))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "game not found")

        async def SubmitBid(self, request, context):
            try:
                game = manager.bid(
                    request.game_id,
                    seat=Seat.parse(request.seat),
                    action=request.action,
                    trump=Suit.parse(request.trump_suit) if request.trump_suit else None,
                    expected_version=request.expected_version or None,
                )
                return game_state(game)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "game not found")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def PlayCard(self, request, context):
            try:
                card: Any = request.card.id if request.card.code == "" else request.card.code
                game = manager.play(
                    request.game_id,
                    seat=Seat.parse(request.seat),
                    card=Card.parse(card),
                    ekka=request.ekka,
                    expected_version=request.expected_version or None,
                )
                return game_state(game)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "game not found")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def ValidateContext(self, request, context):
            await _require_trusted(context)
            try:
                bundle = build_bundle(ROOT, mode=request.rules_profile or "full")
                state = replay_trusted_context(json.loads(request.canonical_json), bundle.rules)
                payload = state.to_dict(reveal_hands=True)
                return platform_pb2.GameState(
                    bundle=bundle_ref(bundle), phase=state.phase.value,
                    current_player=state.current_player.name.lower() if state.current_player is not None else "",
                    canonical_json=_json_bytes(payload),
                    state_sha256=__import__("hashlib").sha256(_json_bytes(payload)).hexdigest(),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def ContinueContext(self, request, context):
            await _require_trusted(context)
            try:
                bundle = build_bundle(ROOT, mode=request.rules_profile or "full")
                state = continue_trusted_context(json.loads(request.canonical_json), bundle.rules)
                payload = state.to_dict(reveal_hands=True)
                return platform_pb2.GameState(
                    bundle=bundle_ref(bundle), phase=state.phase.value,
                    current_player="", canonical_json=_json_bytes(payload),
                    state_sha256=__import__("hashlib").sha256(_json_bytes(payload)).hexdigest(),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def Analyze(self, request, context):
            await _require_trusted(context)
            try:
                bundle = build_bundle(ROOT, mode=request.rules_profile or "full")
                complete = replay_trusted_context(json.loads(request.canonical_json), bundle.rules)
                observer = Seat.parse(request.observer or "south")
                if complete.current_player != observer:
                    raise ValueError("observer must be the current player")
                information = InformationState.from_complete(complete, observer)
                conventions = json.loads(json.dumps(bundle.convention))
                conventions["apply_to_probability"] = bool(request.experimental_conventions)
                result = analyze(
                    information, bundle.rules, conventions,
                    requested_samples=request.samples or 80,
                    exact_allocation_limit=request.exact_allocation_limit or 20000,
                    exact_search_max_remaining=request.exact_search_remaining or 12,
                    seed=request.seed or 1,
                )
                return platform_pb2.AnalysisResult(
                    bundle=bundle_ref(bundle),
                    canonical_json=_json_bytes({"informationState": information.to_dict(), "analysis": result.to_dict()}),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def CreateVideo(self, request, context):
            await _require_trusted(context)
            try:
                payload = json.loads(request.canonical_json)
                validated = VideoJobRequest.model_validate(payload)
                idempotency_key = _metadata(context).get("idempotency-key", "").strip()
                if not idempotency_key:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "idempotency-key metadata is required",
                    )
                job = _submit_legacy_video_compatibility(
                    validated,
                    idempotency_key=idempotency_key,
                )
            except LegacyVideoCompatibilityError as exc:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    json.dumps(exc.diagnostic, sort_keys=True, separators=(",", ":")),
                )
            except SearchTimeBudgetExceeded:
                raise
            except Exception as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return platform_pb2.JobRef(
                job_id=job["jobId"],
                status=job["status"],
                progress=job["progress"],
            )

        async def GetJob(self, request, context):
            tenant = authenticated_rpc.get().tenant_id
            job = STORE.get(request.job_id)
            if job is None:
                if tenant != DEFAULT_TENANT or not request.job_id.startswith("RUN-"):
                    await context.abort(grpc.StatusCode.NOT_FOUND, "job not found")
                await _require_trusted(context)
                run = RESEARCH.get_run(request.job_id)
                if run is None or not _legacy_video_run_is_v4(run):
                    await context.abort(grpc.StatusCode.NOT_FOUND, "job not found")
                job = _legacy_video_run_view(run)
            return platform_pb2.JobResult(
                job=platform_pb2.JobRef(job_id=job["jobId"], status=job["status"], progress=job["progress"]),
                canonical_json=_json_bytes(job),
            )

        async def CreateRenderBatch(self, request, context):
            await _require_trusted(context)
            try:
                items = validate_render_batch_request_payload(request.canonical_json)
                job = RENDER_BATCHES.submit(
                    items,
                    idempotency_key=request.idempotency_key,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return platform_pb2.JobRef(
                job_id=job["jobId"],
                status=job["status"],
                progress=job["progress"],
            )

        async def GetRenderBatch(self, request, context):
            await _require_trusted(context)
            job = RENDER_BATCHES.get(request.job_id)
            if job is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "render batch job not found")
            return platform_pb2.JobResult(
                job=platform_pb2.JobRef(
                    job_id=job["jobId"], status=job["status"], progress=job["progress"]
                ),
                canonical_json=_json_bytes(job),
            )

        async def CancelRenderBatch(self, request, context):
            await _require_trusted(context)
            job = RENDER_BATCHES.get(request.job_id)
            if job is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "render batch job not found")
            if not RENDER_BATCHES.cancel(request.job_id):
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"render batch job cannot be cancelled from {job['status']}",
                )
            updated = RENDER_BATCHES.get(request.job_id)
            return platform_pb2.JobResult(
                job=platform_pb2.JobRef(
                    job_id=updated["jobId"],
                    status=updated["status"],
                    progress=updated["progress"],
                ),
                canonical_json=_json_bytes(updated),
            )

        async def GetReplay(self, request, context):
            operator = authenticated_rpc.get().has_scope(AUTHORIZATION_POLICY, Scope.CONFIG_ADMIN)
            reader = None if operator else (await _identity(context)).subject_id
            game = await _require_private_game(context, request.value, reader)
            game_hash = game.get("gameHash")
            decisions = game.get("decisions") or LEARNING_STORE.list_decisions(game_hash or game.get("gameId"))
            if not operator:
                game = history_without_media({**game, "decisions": decisions})
                decisions = game["decisions"]
            package = {
                "game": game,
                "verification": LEARNING_STORE.verify_game(request.value),
                "audits": LEARNING_STORE.list_game_audits(game_hash or game.get("gameId"), limit=50),
                "videos": LEARNING_STORE.list_video_records(game_hash=game_hash, limit=100) if operator and game_hash else [],
                "decisions": decisions,
            }
            return platform_pb2.ReplayPackage(
                canonical_json=_json_bytes(package),
                game_hash=str(game_hash or ""),
                game_serial=str(game.get("serial") or ""),
            )

        async def ReopenGame(self, request, context):
            await _require_private_game(context, request.identifier)
            try:
                value = LEARNING.reopen_game(request.identifier, branch=bool(request.branch))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "archived game not found")
            game = value.get("game") or {}
            return platform_pb2.HistoryOperationResult(
                canonical_json=_json_bytes(value),
                game_id=str(game.get("gameId") or game.get("id") or ""),
                parent_game_hash=str(value.get("parentGameHash") or ""),
                completed=bool((game.get("state") or {}).get("roundScore")),
            )

        async def ForkDecision(self, request, context):
            await _require_private_decision(context, request.decision_id)
            try:
                action = json.loads(request.action_json) if request.action_json else None
                value = LEARNING.fork_decision(
                    request.decision_id, action=action,
                    continue_to_end=bool(request.continue_to_end),
                    maximum_actions=request.maximum_actions or 128,
                )
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            except (ValueError, json.JSONDecodeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            comparison = value.get("counterfactualComparison") or {}
            return platform_pb2.HistoryOperationResult(
                canonical_json=_json_bytes(value),
                game_id=str(value.get("branchGameId") or value.get("gameId") or ""),
                parent_game_hash=str(value.get("parentGameHash") or ""),
                completed=bool(comparison.get("completed")),
            )

        async def AuditGame(self, request, context):
            await _require_private_game(context, request.identifier)
            try:
                value = AUDIT.audit_game(
                    request.identifier, mode=request.mode or "observer_safe", force=bool(request.rerun),
                    maximum_decisions=request.maximum_decisions or 12,
                    overrides={
                        "requested_samples": request.samples or 96,
                        "exact_allocation_limit": request.exact_allocation_limit or 20000,
                        "exact_search_remaining": request.exact_search_remaining or 10,
                        "use_conventions": bool(request.experimental_conventions),
                        "seed": request.seed or 1,
                    },
                )
                return platform_pb2.AuditResult(
                    canonical_json=_json_bytes(value), audit_id=str(value.get("auditId", "")),
                    report_hash=str(value.get("reportHash", "")),
                )
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "archived game not found")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def ExplainDecision(self, request, context):
            await _require_private_decision(context, request.decision_id)
            try:
                value = AUDIT.audit_decision(
                    request.decision_id, deep=True, force=bool(request.rerun),
                    analysis_id=request.analysis_id or None,
                    overrides={
                        "requested_samples": request.samples or 96,
                        "exact_allocation_limit": request.exact_allocation_limit or 20000,
                        "exact_search_remaining": request.exact_search_remaining or 10,
                        "use_conventions": bool(request.experimental_conventions),
                        "seed": request.seed or 1,
                    },
                )
                return platform_pb2.AuditResult(
                    canonical_json=_json_bytes(value), audit_id=str(value.get("auditId", "")),
                    report_hash=str(value.get("auditHash", "")),
                )
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def CompareDecision(self, request, context):
            await _require_private_decision(context, request.decision_id)
            try:
                value = AUDIT.compare_decision_versions(request.decision_id)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            return platform_pb2.AuditResult(
                canonical_json=_json_bytes(value),
                audit_id=str((value.get("current") or {}).get("auditId", "")),
                report_hash=str((value.get("current") or {}).get("auditHash", "")),
            )

        async def GetDecisionDifficulty(self, request, context):
            await _require_private_decision(context, request.decision_id)
            try:
                _accepted, engine_difficulty, _audit = _accepted_actions_for_decision(request.decision_id)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            value = LEARNING_STORE.get_difficulty_item(request.decision_id) or DIFFICULTY.recalibrate(
                request.decision_id, engine_difficulty=engine_difficulty
            )
            calibrated = float(value.get("calibratedDifficulty", value.get("calibratedScore", engine_difficulty)))
            return platform_pb2.DifficultyResult(
                canonical_json=_json_bytes(value), engine_difficulty=engine_difficulty,
                calibrated_difficulty=calibrated, label=str(value.get("label", "")),
            )

        async def RecordDecisionAttempt(self, request, context):
            await _require_private_decision(context, request.decision_id)
            try:
                chosen = json.loads(request.chosen_action_json)
                accepted, engine_difficulty, _audit = _accepted_actions_for_decision(request.decision_id)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            except json.JSONDecodeError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            value = DIFFICULTY.record_attempt(
                decision_id=request.decision_id, player_id=request.player_id or None, chosen_action=chosen,
                accepted_actions=accepted, engine_difficulty=engine_difficulty,
                response_seconds=request.response_seconds or None, confidence=request.confidence or None,
                reported_skill=request.reported_skill or None, changed_answer=bool(request.changed_answer),
                source="grpc",
            )
            calibrated = float(value.get("calibratedDifficulty", value.get("calibratedScore", engine_difficulty)))
            return platform_pb2.DifficultyResult(
                canonical_json=_json_bytes(value), engine_difficulty=engine_difficulty,
                calibrated_difficulty=calibrated, label=str(value.get("label", "")),
            )

        async def GetChallengeDifficulty(self, request, context):
            challenge = LEARNING_STORE.get_challenge(request.challenge_id)
            if challenge is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "challenge not found")
            accepted, engine_difficulty = _accepted_actions_for_challenge(challenge)
            value = LEARNING_STORE.get_difficulty_item(request.challenge_id) or DIFFICULTY.recalibrate(
                request.challenge_id, engine_difficulty=engine_difficulty
            )
            calibrated = float(value.get("calibratedDifficulty", value.get("calibratedScore", engine_difficulty)))
            return platform_pb2.DifficultyResult(
                canonical_json=_json_bytes({**value, "acceptedActions": accepted}),
                engine_difficulty=engine_difficulty,
                calibrated_difficulty=calibrated,
                label=str(value.get("label", "")),
            )

        async def RecordChallengeAttempt(self, request, context):
            challenge = LEARNING_STORE.get_challenge(request.challenge_id)
            if challenge is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "challenge not found")
            try:
                chosen = json.loads(request.chosen_action_json)
            except json.JSONDecodeError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            accepted, engine_difficulty = _accepted_actions_for_challenge(challenge)
            value = DIFFICULTY.record_attempt(
                decision_id=request.challenge_id,
                player_id=request.player_id or None,
                chosen_action=chosen,
                accepted_actions=accepted,
                engine_difficulty=engine_difficulty,
                response_seconds=request.response_seconds or None,
                confidence=request.confidence or None,
                reported_skill=request.reported_skill or None,
                changed_answer=bool(request.changed_answer),
                source="grpc_challenge",
            )
            calibrated = float(value.get("calibratedDifficulty", value.get("calibratedScore", engine_difficulty)))
            return platform_pb2.DifficultyResult(
                canonical_json=_json_bytes(value),
                engine_difficulty=engine_difficulty,
                calibrated_difficulty=calibrated,
                label=str(value.get("label", "")),
            )

        async def GetAdaptiveChallenge(self, request, context):
            value = DIFFICULTY.adaptive_next(
                request.player_id or None,
                recent_ids=set(request.recent_ids),
            )
            if value is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "no published challenge is available")
            return platform_pb2.AnalysisResult(
                bundle=bundle_ref(build_bundle(ROOT)),
                canonical_json=_json_bytes(value),
            )

        async def GetVideoRecord(self, request, context):
            value = LEARNING_STORE.get_video_record(request.video_id)
            if value is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "video record not found")
            return platform_pb2.VideoRecord(
                canonical_json=_json_bytes(value), video_id=str(value.get("videoId", "")),
                game_hash=str(value.get("gameHash") or ""),
            )

        async def GetVideoTimeline(self, request, context):
            value = LEARNING_STORE.get_video_record(request.video_id)
            if value is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "video record not found")
            timeline = value.get("timeline") or {"segments": []}
            return platform_pb2.VideoRecord(
                canonical_json=_json_bytes({
                    "videoId": value.get("videoId"),
                    "gameHash": value.get("gameHash"),
                    "timeline": timeline,
                }),
                video_id=str(value.get("videoId", "")),
                game_hash=str(value.get("gameHash") or ""),
            )

        async def GetVideoAuditMap(self, request, context):
            try:
                value = build_video_audit_map(LEARNING_STORE, request.video_id)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "video record not found")
            return platform_pb2.VideoRecord(
                canonical_json=_json_bytes(value),
                video_id=str(value.get("videoId", "")),
                game_hash=str(value.get("gameHash") or ""),
            )

        async def RenderGameReplay(self, request, context):
            if not request.identifier:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "identifier is required")
            game = await _require_private_game(context, request.identifier)
            try:
                payload = HistoricalReplayRenderRequest(
                    decisionId=request.decision_id or None,
                    rendererPlugin=request.renderer_plugin or "premium_video",
                    theme=request.theme or "emerald",
                    width=request.width or 1080,
                    height=request.height or 1920,
                    fps=request.fps or 60,
                    crf=request.crf or 18,
                    auditSamples=request.audit_samples or 96,
                    rerunAudit=bool(request.rerun_audit),
                    maximumAuditDecisions=request.maximum_audit_decisions or 40,
                )
            except SearchTimeBudgetExceeded:
                raise
            except Exception as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            job_id = uuid.uuid4().hex[:16]
            STORE.create(
                job_id,
                {"identifier": request.identifier, **payload.model_dump()},
                kind="history_replay_video",
            )
            EXECUTOR.submit(
                _execute_history_replay_render_job,
                job_id,
                request.identifier,
                payload.model_dump(),
            )
            return platform_pb2.JobRef(job_id=job_id, status="queued", progress=0.0)

    class SakkaGrpcService(sakka_pb2_grpc.SakkaPlatformServicer):
        async def _result(self, value: Any):
            return sakka_pb2.JsonResult(canonical_json=_json_bytes(value))

        async def ListBoards(self, request, context):
            identity = await _identity(context)
            return await self._result({"items": SAKKA.store.list_boards(
                subject_id=identity.subject_id, limit=request.limit or 100,
            )})

        async def CreateBoard(self, request, context):
            identity = await _identity(context)
            try:
                data = _payload(request.canonical_json)
                if data.get("source", "new") == "game":
                    board = SAKKA.create_board_from_game(
                        str(data["gameId"]), created_by=identity.subject_id,
                        observer_seat=Seat.parse(str(data.get("observerSeat", "south"))),
                        scope=str(data.get("scope", "round")),
                        maximum_rounds=int(data.get("maximumRounds", 16)),
                        label=data.get("label"), visibility=str(data.get("visibility", "unlisted")),
                    )
                else:
                    board = SAKKA.create_board_from_new_game(
                        created_by=identity.subject_id, seed=int(data.get("seed", 1)),
                        dealer=Seat.parse(str(data.get("dealer", "east"))),
                        mode=str(data.get("rulesProfile", "full")),
                        match_score=(int(data.get("usScore", 0)), int(data.get("themScore", 0))),
                        observer_seat=Seat.parse(str(data.get("observerSeat", "south"))),
                        scope=str(data.get("scope", "round")),
                        maximum_rounds=int(data.get("maximumRounds", 16)), label=data.get("label"),
                        visibility=str(data.get("visibility", "unlisted")),
                        agent_profiles=data.get("agentProfiles"),
                    )
                return await self._result(board.to_dict())
            except (KeyError, ValueError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetBoard(self, request, context):
            identity = await _identity(context)
            item = SAKKA.store.get_board(request.value)
            if item is None or (item[0].visibility == "private" and not owns_subject(identity.subject_id, item[0].created_by)):
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka board not found")
            return await self._result(item[0].to_dict(include_hashes=owns_subject(identity.subject_id, item[0].created_by)))

        async def VerifyBoard(self, request, context):
            identity = await _identity(context)
            item = SAKKA.store.get_board(request.value)
            if item is None or (item[0].visibility == "private" and not owns_subject(identity.subject_id, item[0].created_by)):
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka board not found")
            return await self._result(SAKKA.verify_board(request.value))

        async def CreateRoom(self, request, context):
            identity = await _identity(context)
            try:
                data = _payload(request.canonical_json)
                room = SAKKA.open_room(
                    str(data["boardIdentifier"]), created_by=identity.subject_id,
                    mode=str(data.get("mode", "async")), max_players=int(data.get("maxPlayers", 16)),
                    allow_guests=bool(data.get("allowGuests", True)), settings=data.get("settings") or {},
                )
                return await self._result({"room": room.to_dict(), "shareUrl": f"/play?code={room.room_code}"})
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka board not found")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetRoom(self, request, context):
            try:
                return await self._result(SAKKA.room(request.value))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka room not found")

        async def JoinRoom(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.join_room(
                    request.value, subject_id=identity.subject_id,
                    display_name=identity.display_name, is_guest=identity.is_guest,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka room not found")
            except (PermissionError, ValueError) as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def StartRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.start_run(
                    request.value, subject_id=identity.subject_id,
                    display_name=identity.display_name, is_guest=identity.is_guest,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka room not found")

        async def GetRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.reopen_run(request.value, subject_id=identity.subject_id))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def VerifyRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.verify_run(request.value, subject_id=identity.subject_id))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def PlayAction(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.apply_action(
                    request.run_id, subject_id=identity.subject_id,
                    action=_payload(request.canonical_json),
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def AuditRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.audit_run(
                    request.run_id, subject_id=identity.subject_id,
                    deep=bool(request.deep), force=bool(request.force),
                    maximum_decisions=request.maximum_decisions or 32,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def ShareDecision(self, request, context):
            identity = await _identity(context)
            try:
                data = _payload(request.canonical_json)
                board = SAKKA.create_board_from_run_action(
                    request.run_id, request.sequence, created_by=identity.subject_id,
                    scope=str(data.get("scope", "round")), maximum_rounds=int(data.get("maximumRounds", 16)),
                    label=data.get("label"), visibility=str(data.get("visibility", "unlisted")),
                )
                return await self._result(board.to_dict())
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka decision point not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def RecordFeedback(self, request, context):
            identity = await _identity(context)
            try:
                data = _payload(request.canonical_json)
                return await self._result(SAKKA.feedback(
                    request.run_id, request.sequence, subject_id=identity.subject_id,
                    preferred_action=data.get("preferredAction"),
                    confidence=float(data.get("confidence", 0.8)),
                    rationale=str(data.get("rationale", "")),
                    interestingness=float(data.get("interestingness", 0.0)),
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka decision not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def GetLeaderboard(self, request, context):
            try:
                return await self._result(SAKKA.leaderboard(request.value))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka room not found")

        async def CompareRuns(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.compare_runs(
                    request.run_a, request.run_b,
                    requester_subject_id=identity.subject_id,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Sakka run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


    class SharedGrpcService(shared_pb2_grpc.SharedPlatformServicer):
        """Canonical V9 shared لفة/صكّة API.

        The V8 Sakka service remains registered only as a compatibility alias.
        """

        async def _result(self, value: Any):
            return shared_pb2.JsonResult(canonical_json=_json_bytes(value))

        @staticmethod
        def _scope_payload(board, value: dict[str, Any]) -> dict[str, Any]:
            from baloot_platform.shared import SharedScope
            scope = SharedScope.parse(board.scope)
            value.update({
                "scope": scope.value, "scopeLegacy": scope.legacy, "scopeLabelAr": scope.label_ar,
                "terminology": {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"},
            })
            return value

        async def _owned(self, run_id: str, identity, context):
            from baloot_platform.storage.codec_v7 import decode
            loaded = SAKKA.store.get_run(run_id)
            if loaded is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared run not found")
            run, payload = loaded
            if not owns_subject(identity.subject_id, run.subject_id):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "run belongs to another player")
            board_item = SAKKA.store.get_board(run.board_id)
            room = SAKKA.store.get_room(run.room_id)
            if board_item is None or room is None:
                await context.abort(grpc.StatusCode.INTERNAL, "shared run references missing resources")
            return run, payload, board_item[0], room, decode(payload, root=ROOT)

        async def ListBoards(self, request, context):
            from baloot_platform.shared import SharedScope
            identity = await _identity(context)
            items = SAKKA.store.list_boards(subject_id=identity.subject_id, limit=request.limit or 100)
            for item in items:
                scope = SharedScope.parse(item.get("scope", "round"))
                item.update({"scope": scope.value, "scopeLegacy": scope.legacy, "scopeLabelAr": scope.label_ar})
            return await self._result({"items": items, "terminology": {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"}})

        async def CreateBoard(self, request, context):
            from baloot_platform.shared import SharedScope
            identity = await _identity(context)
            try:
                data = _payload(request.canonical_json)
                scope = SharedScope.parse(str(data.get("scope", "laffa")))
                if data.get("source", "new") == "game":
                    board = SAKKA.create_board_from_game(
                        str(data["gameId"]), created_by=identity.subject_id,
                        observer_seat=Seat.parse(str(data.get("observerSeat", "south"))),
                        scope=scope.value, maximum_rounds=int(data.get("maximumRounds", 16)),
                        label=data.get("label"), visibility=str(data.get("visibility", "unlisted")),
                    )
                else:
                    board = SAKKA.create_board_from_new_game(
                        created_by=identity.subject_id, seed=int(data.get("seed", 1)),
                        dealer=Seat.parse(str(data.get("dealer", "east"))),
                        mode=str(data.get("rulesProfile", "full")),
                        match_score=(int(data.get("usScore", 0)), int(data.get("themScore", 0))),
                        observer_seat=Seat.parse(str(data.get("observerSeat", "south"))),
                        scope=scope.value, maximum_rounds=int(data.get("maximumRounds", 16)),
                        label=data.get("label"), visibility=str(data.get("visibility", "unlisted")),
                        agent_profiles=data.get("agentProfiles"),
                    )
                value = board.to_dict(include_hashes=True)
                return await self._result(self._scope_payload(board, value))
            except (KeyError, ValueError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetBoard(self, request, context):
            identity = await _identity(context)
            item = SAKKA.store.get_board(request.value)
            if item is None or (item[0].visibility == "private" and not owns_subject(identity.subject_id, item[0].created_by)):
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared board not found")
            board = item[0]
            return await self._result(self._scope_payload(board, board.to_dict(include_hashes=owns_subject(identity.subject_id, board.created_by))))

        async def VerifyBoard(self, request, context):
            identity = await _identity(context)
            item = SAKKA.store.get_board(request.value)
            if item is None or (item[0].visibility == "private" and not owns_subject(identity.subject_id, item[0].created_by)):
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared board not found")
            return await self._result(SAKKA.verify_board(request.value))

        async def CreateRoom(self, request, context):
            identity = await _identity(context)
            try:
                data = _payload(request.canonical_json)
                room = SAKKA.open_room(
                    str(data["boardIdentifier"]), created_by=identity.subject_id,
                    mode=str(data.get("mode", "async")), max_players=int(data.get("maxPlayers", 16)),
                    allow_guests=bool(data.get("allowGuests", True)), settings=data.get("settings") or {},
                )
                return await self._result({"room": room.to_dict(), "shareUrl": f"/play?code={room.room_code}"})
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared board not found")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetRoom(self, request, context):
            try:
                return await self._result(SAKKA.room(request.value))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared room not found")

        async def JoinRoom(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.join_room(request.value, subject_id=identity.subject_id, display_name=identity.display_name, is_guest=identity.is_guest))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared room not found")
            except (PermissionError, ValueError) as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def StartRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.start_run(request.value, subject_id=identity.subject_id, display_name=identity.display_name, is_guest=identity.is_guest))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared room not found")
            except (PermissionError, ValueError) as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def GetRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.reopen_run(request.value, subject_id=identity.subject_id))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def VerifyRun(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.verify_run(request.value, subject_id=identity.subject_id))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def PlayAction(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(SAKKA.apply_action(request.run_id, subject_id=identity.subject_id, action=_payload(request.canonical_json)))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetRunReview(self, request, context):
            identity = await _identity(context)
            run, _payload_raw, board, _room, session = await self._owned(request.run_id, identity, context)
            actions = SAKKA.store.actions(request.run_id)
            seq_by_decision = {str(item.get("decisionId")): int(item["sequence"]) for item in actions if item.get("decisionId")}
            decisions = LEARNING_STORE.list_decisions(session.id, limit=10_000) if LEARNING_STORE else []
            completed = run.status == "completed"
            rows=[]
            for decision in decisions:
                actor = str(decision.get("actor") or "")
                relation = observer_relation(actor, board.observer_seat)
                can_review = private_decision_review_allowed(
                    actor=actor,
                    observer=board.observer_seat,
                    run_completed=completed,
                )
                explanation = EXPLANATIONS.explain(
                    str(decision["decisionId"]), observer=board.observer_seat.name.lower(), level="basic",
                    surface="replay", run_completed=completed, challenge_revealed=completed,
                    view_mode="observer", ensure_audit=bool(request.ensure_audit and can_review),
                )
                rows.append({
                    "decisionId": decision["decisionId"], "ordinal": decision["ordinal"],
                    "decisionKind": decision["decisionKind"], "actor": decision["actor"],
                    "actorRelation": relation, "selectedAction": decision.get("selectedAction"),
                    "legalActions": (decision.get("legalActions") or []) if can_review else [],
                    "actionSequence": seq_by_decision.get(str(decision["decisionId"])),
                    "reviewable": can_review, "simpleExplanation": explanation,
                    "audits": (decision.get("audits") or []) if can_review else [],
                    "reviewConsensus": (decision.get("reviewConsensus") or {}) if can_review else {},
                })
            return await self._result({
                "runId": request.run_id, "status": run.status, "scope": board.scope,
                "boardId": board.board_id, "boardSerial": board.serial, "boardHash": board.board_hash,
                "environmentHash": board.environment_hash, "decisions": rows,
                "terminology": {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"},
            })

        async def ExplainDecision(self, request, context):
            identity = await _identity(context)
            run, _payload_raw, board, _room, session = await self._owned(request.run_id, identity, context)
            decision = LEARNING_STORE.get_decision(request.decision_id) if LEARNING_STORE else None
            if decision is None or str(decision.get("gameId")) != str(session.id):
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            completed = run.status == "completed"
            view_mode = request.view_mode or "observer"
            if view_mode == "oracle_after_completion" and not completed:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "oracle explanation is available after completion only")
            return await self._result(EXPLANATIONS.explain(
                request.decision_id, observer=board.observer_seat.name.lower(), level=request.level or "basic",
                surface=request.surface or "replay", run_completed=completed,
                challenge_revealed=bool(request.challenge_revealed and completed), view_mode=view_mode,
                ensure_audit=bool(request.ensure_audit),
            ))

        async def GetDecisionReviews(self, request, context):
            identity = await _identity(context)
            run, _payload_raw, board, _room, session = await self._owned(request.run_id, identity, context)
            decision = LEARNING_STORE.get_decision(request.decision_id) if LEARNING_STORE else None
            if decision is None or str(decision.get("gameId")) != str(session.id):
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            if not private_decision_review_allowed(
                actor=str(decision.get("actor") or ""),
                observer=board.observer_seat,
                run_completed=run.status == "completed",
            ):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "other players' reviews are available after completion")
            value = REVIEWS.context(request.decision_id, run_id=request.run_id, view_mode="observer")
            consensus = dict(value.get("consensus") or {})
            consensus["agreement"] = consensus.get(
                "supportShare", consensus.get("agreement", 0.0)
            )
            value["consensus"] = consensus
            return await self._result(value)

        async def SubmitDecisionReview(self, request, context):
            identity = await _identity(context)
            run, _payload_raw, board, _room, session = await self._owned(request.run_id, identity, context)
            decision = LEARNING_STORE.get_decision(request.decision_id) if LEARNING_STORE else None
            if decision is None or str(decision.get("gameId")) != str(session.id):
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")
            if not private_decision_review_allowed(
                actor=str(decision.get("actor") or ""),
                observer=board.observer_seat,
                run_completed=run.status == "completed",
            ):
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "other players' decisions can be reviewed after completion")
            try:
                data=_payload(request.canonical_json)
                value=REVIEWS.submit(
                    request.decision_id, reviewer_id=identity.subject_id, reviewer_role="player",
                    preferred_action=data.get("preferredAction"), confidence=float(data.get("confidence",0.8)),
                    reason_codes=data.get("reasonCodes") or ["other"], rationale=str(data.get("rationale","")),
                    evidence=data.get("evidence") or [], view_mode=str(data.get("viewMode","observer")),
                    source="grpc_decision_workbench",
                )
                consensus = dict(value.get("consensus") or {})
                consensus["agreement"] = consensus.get(
                    "supportShare", consensus.get("agreement", 0.0)
                )
                value["consensus"] = consensus
                return await self._result(value)
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def ShareDecision(self, request, context):
            identity = await _identity(context)
            try:
                data=_payload(request.canonical_json)
                board=SAKKA.create_board_from_run_action(
                    request.run_id, request.sequence, created_by=identity.subject_id,
                    scope=str(data.get("scope","laffa")), maximum_rounds=int(data.get("maximumRounds",16)),
                    label=data.get("label"), visibility=str(data.get("visibility","unlisted")),
                )
                return await self._result(self._scope_payload(board, board.to_dict(include_hashes=True)))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision point not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetLeaderboard(self, request, context):
            try:
                return await self._result(SAKKA.leaderboard(request.value))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared room not found")

        async def CompareRuns(self, request, context):
            identity = await _identity(context)
            try:
                value=SAKKA.compare_runs(request.run_a, request.run_b, requester_subject_id=identity.subject_id)
                value["terminology"]={"matchAr":"صكّة","roundAr":"لفة","trickAr":"أكلة"}
                return await self._result(value)
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "shared run not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


    class HumanLearningService(hitl_pb2_grpc.HumanLearningServicer):
        @staticmethod
        async def _result(value: dict[str, Any]):
            return hitl_pb2.JsonResponse(canonical_json=_json_bytes(value))

        async def HealthCheck(self, request, context):
            return hitl_pb2.Health(status="ok", version=__version__)

        async def Profile(self, request, context):
            await _identity(context)
            return await self._result({"profile": HITL_PROFILE, "reasonTaxonomy": list(HITL.REASON_TAXONOMY),
                                       "automaticProductionApply": False, "automaticModelPromotion": False})

        async def ListCases(self, request, context):
            await _identity(context); data = _payload(request.canonical_json)
            return await self._result({"items": HITL.list_cases(limit=int(data.get("limit", 100)))})

        async def CreateCase(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                return await self._result(HITL.create_case(str(data["decisionId"]), created_by=identity.subject_id,
                    high_impact=bool(data.get("highImpact", False)), completed=bool(data.get("completed", True))))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "decision not found")

        async def GetCase(self, request, context):
            await _identity(context); data = _payload(request.canonical_json)
            try: return await self._result(HITL.case(str(data["caseId"]), blind=bool(data.get("blind", False))))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "review case not found")

        async def AssignBlindReview(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                return await self._result(HITL.assign(str(data["caseId"]), reviewer_id=identity.subject_id,
                    device_id=str(data.get("deviceId", "")), network_id=str(data.get("networkId", "")),
                    ttl_hours=int(data.get("ttlHours", 72))))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "review case not found")

        async def SubmitReview(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                return await self._result(HITL.submit(str(data["caseId"]), assignment_id=str(data["assignmentId"]),
                    assignment_token=str(data["assignmentToken"]), reviewer_id=identity.subject_id,
                    preferred_action=data.get("preferredAction"), confidence=float(data.get("confidence", .5)),
                    reason_codes=list(data.get("reasonCodes") or ["other"]), rationale_ar=str(data.get("rationaleAr", "")),
                    claims=list(data.get("claims") or []), evidence=list(data.get("evidence") or []),
                    view_mode=str(data.get("viewMode", "observer")), hidden_cards_seen=bool(data.get("hiddenCardsSeen", False)),
                    final_result_seen=bool(data.get("finalResultSeen", False)), disclosure_timing=str(data.get("disclosureTiming", "before_reveal"))))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "review case not found")
            except ValueError as exc: await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def ComputeConsensus(self, request, context):
            await _identity(context)
            try: return await self._result(HITL.compute_consensus(request.value))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "review case not found")

        async def SetConsent(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            mapping = {"allow_training": bool(data.get("allowTraining", False)), "allow_research": bool(data.get("allowResearch", False)),
                       "allow_public_challenge": bool(data.get("allowPublicChallenge", False)), "allow_voice": bool(data.get("allowVoice", False)),
                       "allow_identity_attribution": bool(data.get("allowIdentityAttribution", False)),
                       "allow_private_coaching": bool(data.get("allowPrivateCoaching", True)), "revoke": bool(data.get("revoke", False))}
            return await self._result(HITL.set_consent(identity.subject_id, **mapping))

        async def GetConsent(self, request, context):
            identity = await _identity(context)
            subject = request.value or identity.subject_id
            if subject != identity.subject_id and identity.kind not in {"admin", "service", "referee"}:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, "consent is private")
            return await self._result(HITL.get_consent(subject))

        async def CreateTrainingCandidate(self, request, context):
            await _identity(context)
            try: return await self._result(HITL.create_training_candidate(request.value))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "review case not found")
            except ValueError as exc: await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def RecordChallengerEvaluation(self, request, context):
            await _identity(context); data = _payload(request.canonical_json)
            try: return await self._result(HITL.record_challenger_evaluation(str(data["candidateId"]),
                validator_id=str(data["validatorId"]), validator_type=str(data.get("validatorType", "independent_solver")),
                status=str(data["status"]), metrics=dict(data.get("metrics") or {}), independent=bool(data.get("independent", True))))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "training candidate not found")

        async def AcceptTrainingCandidate(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try: return await self._result(HITL.accept_training_candidate(str(data["candidateId"]), accepted_by=identity.subject_id))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "training candidate not found")
            except ValueError as exc: await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def CreateImprovementProposal(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try: return await self._result(HITL.create_improvement_proposal(created_by=identity.subject_id,
                target=str(data["target"]), title_ar=str(data["titleAr"]), rationale_ar=str(data["rationaleAr"]),
                source_case_ids=list(data.get("sourceCaseIds") or []), requested_change=dict(data.get("requestedChange") or {})))
            except ValueError as exc: await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def EvaluateImprovement(self, request, context):
            await _identity(context); data = _payload(request.canonical_json)
            try: return await self._result(HITL.evaluate_improvement(str(data["proposalId"]), validator_id=str(data["validatorId"]),
                status=str(data["status"]), evidence=dict(data.get("evidence") or {}), independent=bool(data.get("independent", True))))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "proposal not found")

        async def ApproveImprovement(self, request, context):
            identity = await _identity(context)
            auth = authenticated_rpc.get()
            data = _payload(request.canonical_json)
            role = identity.kind.strip().lower()
            if auth.api_key_valid and auth.identity is None and auth.claims is None:
                # Key-only callers retain the legacy delegated-role contract.
                role = str(data.get("role") or identity.kind)
            try:
                return await self._result(HITL.approve_improvement(
                    str(data["proposalId"]), approved_by=identity.subject_id, role=role,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "proposal not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def FinalizeImprovement(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try: return await self._result(HITL.finalize_improvement(str(data["proposalId"]), release_governor_id=identity.subject_id,
                change_manifest=dict(data.get("changeManifest") or {})))
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "proposal not found")
            except ValueError as exc: await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def RenderReviewClip(self, request, context):
            await _identity(context); data = _payload(request.canonical_json)
            try:
                case_id = str(data["caseId"]); artifacts = HITL_REVIEW_RENDERER.render(case=HITL.case(case_id, blind=True),
                    output_dir=OUTPUTS, basename=f"baloot-hitl-{case_id}", width=data.get("width"), height=data.get("height"),
                    fps=data.get("fps"), duration=data.get("durationSeconds"))
                return await self._result({"video": str(artifacts.video), "poster": str(artifacts.poster),
                                           "timeline": str(artifacts.timeline), "manifest": str(artifacts.manifest)})
            except KeyError: await context.abort(grpc.StatusCode.NOT_FOUND, "review case not found")
            except (ValueError, RuntimeError) as exc: await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


    class OperationsService(operations_pb2_grpc.BalootOperationsServicer):
        @staticmethod
        async def _result(value: dict[str, Any]):
            return operations_pb2.JsonResponse(canonical_json=_json_bytes(value))

        async def HealthCheck(self, request, context):
            return operations_pb2.Health(status="ok", version=__version__)

        async def CreateLiveRoom(self, request, context):
            identity = await _identity(context)
            data = _payload(request.canonical_json)
            try:
                if data.get("gameId"):
                    game = LIVE.games.get(str(data["gameId"]))
                else:
                    game = LIVE.games.create(
                        seed=int(data.get("seed", 1)), dealer=Seat.parse(str(data.get("dealer", "east"))),
                        mode=str(data.get("rulesProfile", "full")),
                        match_score=(int(data.get("usScore", 0)), int(data.get("themScore", 0))),
                    )
                room = LIVE.create_room(
                    game_id=game.id, owner_subject_id=identity.subject_id,
                    turn_seconds=int(data.get("turnSeconds", 45)),
                    reconnect_grace_seconds=int(data.get("reconnectGraceSeconds", 120)),
                    allow_spectators=bool(data.get("allowSpectators", True)),
                    assistance_policy=str(data.get("assistancePolicy", "none")),
                )
                return await self._result({"room": room.to_dict(include_events=False), "joinCode": room.room_code})
            except (ValueError, KeyError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetLiveRoom(self, request, context):
            identity = await _identity(context)
            try:
                return await self._result(LIVE.view(request.value, subject_id=identity.subject_id))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "live room not found")

        async def ClaimLiveSeat(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                return await self._result(LIVE.claim_seat(
                    str(data["identifier"]), subject_id=identity.subject_id,
                    display_name=str(data.get("displayName") or identity.display_name), seat=str(data["seat"]),
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "live room not found")
            except (ValueError, PermissionError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def SetLiveReady(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                room = LIVE.set_ready(str(data["identifier"]), subject_id=identity.subject_id, ready=bool(data.get("ready", True)))
                return await self._result({"room": room.to_dict(include_events=False)})
            except (KeyError, ValueError, PermissionError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def StartLiveTable(self, request, context):
            identity = await _identity(context)
            try:
                room = LIVE.start(request.value, requested_by=identity.subject_id)
                return await self._result({"room": room.to_dict(include_events=False)})
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "live room not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def SubmitLiveAction(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                return await self._result(LIVE.submit_action(
                    str(data["identifier"]), subject_id=identity.subject_id, action=dict(data["action"]),
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "live room not found")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def VerifyLiveRoom(self, request, context):
            await _identity(context)
            try:
                return await self._result(LIVE.verify(request.value))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "live room not found")

        async def OpenAdjudicationCase(self, request, context):
            identity = await _identity(context); data = _payload(request.canonical_json)
            try:
                case = ADJUDICATION.open_case(
                    game_id=str(data["gameId"]), violation=str(data["violation"]),
                    reported_by=identity.subject_id, description=str(data["description"]),
                    round_number=data.get("roundNumber"), event_sequence=data.get("eventSequence"),
                    accused_subject_id=data.get("accusedSubjectId"), accused_seat=data.get("accusedSeat"),
                    metadata=data.get("metadata") or {},
                )
                return await self._result(case.to_dict())
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetAdjudicationCase(self, request, context):
            await _identity(context); case = ADJUDICATION.store.get(request.value)
            if case is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "adjudication case not found")
            return await self._result(case.to_dict())

        async def ListAdjudicationCases(self, request, context):
            await _identity(context); data = _payload(request.canonical_json)
            return await self._result({"items": ADJUDICATION.store.list(
                game_id=data.get("gameId"), status=data.get("status"), limit=int(data.get("limit", 100)),
            )})

        async def RecommendAdjudicationRuling(self, request, context):
            await _identity(context)
            try:
                return await self._result(ADJUDICATION.recommended_ruling(request.value))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "adjudication case not found")

        async def SecurityCapabilities(self, request, context):
            return await self._result({
                "optionalSignIn": True, "localGuest": True, "localAccount": True,
                "passkeys": True, "scopes": sorted({scope for values in AUTHORIZATION_POLICY.role_scopes.values() for scope in values}),
            })


    class ConfigurationServiceV19(configuration_pb2_grpc.ConfigurationControlServicer):
        @staticmethod
        def _result(value: dict[str, Any]):
            raw = _json_bytes(value)
            return configuration_pb2.JsonResult(canonical_json=raw, sha256=__import__("hashlib").sha256(raw).hexdigest())

        @staticmethod
        def _tenant(context) -> str | None:
            value = _metadata(context).get("x-baloot-tenant-id", "").strip()
            return value or None

        async def GetCapabilities(self, request, context):
            await _identity(context)
            return self._result({
                "rowVersioned": True, "tenantAware": True, "contentAddressed": True,
                "optimisticConcurrency": True, "automaticProductionApply": False,
                "profile": CONFIGURATION_PROFILE, "performance": PERFORMANCE_PROFILE,
            })

        async def ListProfiles(self, request, context):
            await _identity(context)
            values = CONFIGURATION.store.list_profiles(
                tenant_id=request.tenant_id or self._tenant(context), family=request.family or None,
                profile_id=request.profile_id or None, status=request.status or None,
                scope=request.scope or None, limit=request.limit or 200,
            )
            return self._result({"items": [item.to_dict(include_payload=False) for item in values]})

        async def CreateProfile(self, request, context):
            identity = await _identity(context)
            try:
                value = CONFIGURATION.create_profile(
                    family=request.family, payload=_payload(request.canonical_json), created_by=identity.subject_id,
                    scope=request.scope or "global", tenant_id=request.tenant_id or self._tenant(context),
                    supersedes_revision_id=request.supersedes_revision_id or None, publish=request.publish,
                    require_matching_schedule=False,
                )
                return self._result(value)
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))

        async def PublishProfile(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.publish_profile(
                    request.value, expected_etag=request.expected_etag or 1, actor_id=identity.subject_id,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "profile revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def RejectProfile(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.reject_profile(request.value, expected_etag=request.expected_etag or 1, actor_id=identity.subject_id, reason=request.reason))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "profile revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def RetireProfile(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.retire_profile(request.value, expected_etag=request.expected_etag or 1, actor_id=identity.subject_id, reason=request.reason))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "profile revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def ResolveProfile(self, request, context):
            await _identity(context)
            try:
                return self._result(CONFIGURATION.resolve_profile(
                    family=request.family, profile_id=request.profile_id,
                    tenant_id=request.tenant_id or self._tenant(context),
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "profile not resolved")

        async def CompareProfiles(self, request, context):
            await _identity(context)
            left = CONFIGURATION.store.get_profile(request.left); right = CONFIGURATION.store.get_profile(request.right)
            if left is None or right is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, "profile revision not found")
            return self._result({
                "left": left.to_dict(include_payload=False), "right": right.to_dict(include_payload=False),
                "changes": CONFIGURATION.compare_payloads(dict(left.payload), dict(right.payload)),
            })

        async def ListBundles(self, request, context):
            await _identity(context)
            values = CONFIGURATION.store.list_bundles(
                tenant_id=request.tenant_id or self._tenant(context), bundle_id=request.bundle_id or None,
                status=request.status or None, scope=request.scope or None, limit=request.limit or 200,
                include_scheduled=True,
            )
            return self._result({"items": [item.to_dict() for item in values]})

        async def CreateBundle(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.create_bundle(
                    bundle_id=request.bundle_id, version=request.version, mode=request.mode or "full",
                    component_revisions=dict(request.components), created_by=identity.subject_id,
                    scope=request.scope or "global", tenant_id=request.tenant_id or self._tenant(context),
                    supersedes_revision_id=request.supersedes_revision_id or None,
                    publish=request.publish, allow_draft_components=request.allow_drafts,
                    require_matching_schedule=False,
                ))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))

        async def PublishBundle(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.publish_bundle(
                    request.value, expected_etag=request.expected_etag or 1, actor_id=identity.subject_id,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "bundle revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def RejectBundle(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.reject_bundle(request.value, expected_etag=request.expected_etag or 1, actor_id=identity.subject_id, reason=request.reason))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "bundle revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def RetireBundle(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.retire_bundle(request.value, expected_etag=request.expected_etag or 1, actor_id=identity.subject_id, reason=request.reason))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "bundle revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        async def ResolveBundle(self, request, context):
            await _identity(context)
            try:
                return self._result(CONFIGURATION.resolve(
                    revision_id=request.revision_id or None, bundle_id=request.bundle_id or None,
                    tenant_id=request.tenant_id or self._tenant(context), allow_draft=request.allow_draft,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "bundle not resolved")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def ResolveEffectiveRuntimeConfigV2(self, request, context):
            await _identity(context)
            try:
                return self._result(CONFIGURATION.freeze_v2(
                    revision_id=request.revision_id or None,
                    bundle_id=request.bundle_id or None,
                    tenant_id=request.tenant_id or self._tenant(context),
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "bundle not resolved")
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        async def RollbackBundle(self, request, context):
            identity = await _identity(context)
            try:
                return self._result(CONFIGURATION.rollback_bundle(
                    request.revision_id,
                    new_version=request.new_version,
                    actor_id=identity.subject_id,
                ))
            except KeyError:
                await context.abort(grpc.StatusCode.NOT_FOUND, "bundle revision not found")
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.ABORTED, str(exc))
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def ListPlugins(self, request, context):
            await _identity(context)
            return self._result({"items": REGISTRY.list(request.kind or None), "externalInProcessAllowed": False, "policy": REGISTRY.policy.to_dict()})

        async def GetPerformanceProfile(self, request, context):
            await _identity(context)
            return self._result(PERFORMANCE_PROFILE)

        async def RunPerformanceQuick(self, request, context):
            await _identity(context)
            values = [
                PERFORMANCE.measure("configuration_resolve_p95_ms", CONFIGURATION.resolve, iterations=50, warmup=5),
                PERFORMANCE.measure("plugin_registry_list_p95_ms", lambda: REGISTRY.list(), iterations=100, warmup=10),
            ]
            return self._result(PERFORMANCE.report(values))

        async def GetCapabilityMatrix(self, request, context):
            await _identity(context)
            return self._result(CAPABILITY_MATRIX)

        async def GetCertificationProfile(self, request, context):
            await _identity(context)
            return self._result({"profile": CERTIFICATION_PROFILE, "automaticRuleActivation": False})

        async def VerifyCertificationCandidate(self, request, context):
            await _identity(context)
            try:
                return self._result(verify_certification_candidate(_payload(request.canonical_json).get("candidate", _payload(request.canonical_json))))
            except (ValueError, TypeError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetBenchmarkProfile(self, request, context):
            await _identity(context)
            return self._result({"profile": BENCHMARK_PROFILE, "automaticPromotion": False})

        async def EvaluateIntelligenceBenchmark(self, request, context):
            await _identity(context)
            value = _payload(request.canonical_json)
            try:
                result = IntelligenceBenchmarkV18(BENCHMARK_PROFILE).evaluate(str(value.get("modelId") or value.get("model_id") or "candidate"), list(value.get("examples") or []))
                return self._result(result.to_dict())
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def GetOperationsProfile(self, request, context):
            await _identity(context)
            return self._result({"profile": OPERATIONS_PROFILE, "automaticProductionChange": False})


    class AuthorizationInterceptor(grpc.aio.ServerInterceptor):
        async def intercept_service(self, continuation, handler_call_details):
            handler = await continuation(handler_call_details)
            method = handler_call_details.method
            if handler is None or handler.unary_unary is None:
                return None

            async def invoke(request, context):
                try:
                    scope = rpc_scope(method)
                except KeyError:
                    return await context.abort(
                        grpc.StatusCode.PERMISSION_DENIED, "RPC authorization policy is unavailable",
                    )
                if scope is None:
                    return await handler.unary_unary(request, context)
                metadata = _metadata(context)
                authorization = metadata.get("authorization", "")
                token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else None
                auth = credentials(metadata, token, IDENTITY, ACCESS_TOKENS)
                if not auth.authenticated or (
                    method in RPC_SERVICE_KEY_METHODS and not auth.api_key_valid
                ):
                    await context.abort(grpc.StatusCode.UNAUTHENTICATED, "authentication required")
                denial = auth.denial(AUTHORIZATION_POLICY, scope, tenant_job=tenant_job_rpc(method))
                if denial is not None:
                    await context.abort(grpc.StatusCode.PERMISSION_DENIED, denial)
                auth_token = authenticated_rpc.set(auth)
                ownership_token = SERVICE_SUBJECT_OWNER.set(
                    service_subject_ownership(metadata, configured_key=os.environ.get("BALOOT_API_KEY"))
                    if auth.api_key_valid and auth.identity is None and auth.claims is None else None
                )
                try:
                    identity = await _identity(context)
                    database_token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(
                        subject_id=identity.subject_id, tenant_id=auth.tenant_id, role=auth.role,
                        session_id=str((auth.session or {}).get("session_id") or
                                       (auth.claims.session_id if auth.claims else "") or ""),
                    ))
                    try:
                        return await handler.unary_unary(request, context)
                    finally:
                        CURRENT_DATABASE_IDENTITY.reset(database_token)
                finally:
                    SERVICE_SUBJECT_OWNER.reset(ownership_token)
                    authenticated_rpc.reset(auth_token)

            return grpc.unary_unary_rpc_method_handler(
                invoke, request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

    class ApplicationErrorInterceptor(grpc.aio.ServerInterceptor):
        async def intercept_service(self, continuation, handler_call_details):
            handler = await continuation(handler_call_details)
            # shortcut: current RPCs are unary-only, wrap streams when one is added.
            if handler is None or handler.unary_unary is None:
                return handler

            async def invoke(request, context):
                try:
                    return await handler.unary_unary(request, context)
                except SearchTimeBudgetExceeded as exc:
                    await context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        json.dumps(exc.to_detail(), sort_keys=True, separators=(",", ":")),
                    )
                except ConfigurationScheduleConflict as exc:
                    await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
                except ConfigurationIntegrityError:
                    await context.abort(
                        grpc.StatusCode.UNAVAILABLE,
                        "Configuration is unavailable. Ask a configuration administrator to restore a verified revision.",
                    )

            return grpc.unary_unary_rpc_method_handler(
                invoke, request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

    server = grpc.aio.server(interceptors=[AuthorizationInterceptor(), ApplicationErrorInterceptor()], options=[
        ("grpc.max_receive_message_length", 2 * 1024 * 1024),
        ("grpc.max_send_message_length", 8 * 1024 * 1024),
    ])
    platform_pb2_grpc.add_BalootPlatformServicer_to_server(Service(), server)
    sakka_pb2_grpc.add_SakkaPlatformServicer_to_server(SakkaGrpcService(), server)
    shared_pb2_grpc.add_SharedPlatformServicer_to_server(SharedGrpcService(), server)
    operations_pb2_grpc.add_BalootOperationsServicer_to_server(OperationsService(), server)
    hitl_pb2_grpc.add_HumanLearningServicer_to_server(HumanLearningService(), server)
    configuration_pb2_grpc.add_ConfigurationControlServicer_to_server(ConfigurationServiceV19(), server)
    bind = os.environ.get("BALOOT_GRPC_BIND", "[::]:50051")
    if os.environ.get("BALOOT_GRPC_TLS_CERT") and os.environ.get("BALOOT_GRPC_TLS_KEY"):
        cert = Path(os.environ["BALOOT_GRPC_TLS_CERT"]).read_bytes()
        key = Path(os.environ["BALOOT_GRPC_TLS_KEY"]).read_bytes()
        server.add_secure_port(bind, grpc.ssl_server_credentials(((key, cert),)))
    else:
        server.add_insecure_port(bind)
    await server.start()
    await server.wait_for_termination()


def main() -> None:  # pragma: no cover
    configure_event_logging()
    try:
        asyncio.run(serve())
    except ConfigurationIntegrityError:
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
