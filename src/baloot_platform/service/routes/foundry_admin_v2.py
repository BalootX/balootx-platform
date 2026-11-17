"""Thin V2 admin resources over authoritative Foundry application services."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Literal, Mapping

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from baloot_platform.learning.challenge_publication import validate_reviewer_role
from baloot_platform.service.laffa_library import LaffaLibraryServiceV1

from .auth import require_identity


class CorrectionCreateV2(BaseModel):
    decisionId: str = Field(min_length=1, max_length=200)
    correctionType: str = Field(default="analysis_defect", min_length=1, max_length=100)
    highImpact: bool = False
    completed: bool = True


class DatasetCreateV2(BaseModel):
    purpose: str = Field(default="human_corrected_play_v1", min_length=1, max_length=200)
    minimumConfidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelCreateV2(BaseModel):
    datasetId: str = Field(min_length=1, max_length=200)
    backend: Literal["auto", "numpy", "torch"] = "auto"
    epochs: int | None = Field(default=None, ge=1, le=10_000)
    learningRate: float | None = Field(default=None, gt=0.0, le=10.0)
    l2: float | None = Field(default=None, ge=0.0, le=10.0)
    batchSize: int | None = Field(default=None, ge=1, le=100_000)
    seed: int | None = None


class PublicationTransitionV2(BaseModel):
    challengeId: str = Field(min_length=1, max_length=200)
    action: Literal["review", "reject", "approve", "publish", "retire"]
    evidenceSha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class CompleteTraceReferenceV2(BaseModel):
    ownerKind: Literal["complete_trace_v1"]
    ownerHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str = Field(min_length=1, max_length=2000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)
    mediaType: Literal["application/vnd.balootx.complete-trace-v1+json"]
    name: str | None = Field(default=None, max_length=500)


class ContextSearchCreateV2(BaseModel):
    parentTraceReference: CompleteTraceReferenceV2
    decisionId: str = Field(min_length=1, max_length=200)
    contextProofHashes: list[str] = Field(min_length=1, max_length=10_000)


class NaturalSakkaCreateV2(BaseModel):
    target: int = Field(default=152, ge=1, le=10_000)
    traceReferences: list[CompleteTraceReferenceV2] = Field(min_length=1, max_length=1_000)


class CuratedSakkaSelectionV2(BaseModel):
    contextSearchResultHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contextualizedLaffaHash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CuratedSakkaCreateV2(BaseModel):
    target: int = Field(default=152, ge=1, le=10_000)
    selections: list[CuratedSakkaSelectionV2] = Field(min_length=1, max_length=1_000)


class NaturalSakkaGenerationCreateV2(BaseModel):
    rootSeed: int = Field(ge=0, le=2**63 - 1)
    initialDealer: Literal["north", "east", "south", "west"] = "east"


class LaffaSourceReferenceV1(BaseModel):
    kind: Literal[
        "archived_game",
        "sakka_round",
        "research_result",
        "qualified_laffa",
        "research_audit_trace",
    ]
    code: str
    identity: str
    uri: str


class SakkaReferenceV1(BaseModel):
    code: str
    identity: str
    uri: str


class UnifiedLaffaRowV1(BaseModel):
    humanCode: str
    sourceKind: Literal[
        "archived_game",
        "sakka_round",
        "research_result",
        "qualified_laffa",
        "research_audit_trace",
    ]
    sourceKinds: list[
        Literal[
            "archived_game",
            "sakka_round",
            "research_result",
            "qualified_laffa",
            "research_audit_trace",
        ]
    ]
    replayValid: bool
    auditOnly: bool
    publicationEligible: bool
    qualificationFailures: list[str]
    qualificationStatus: str
    truthLabel: str | None = None
    confidence: float | None = None
    challengeScore: float | None = None
    tacticalTags: list[str]
    contract: str | None = None
    project: str | None = None
    projects: list[str]
    buyer: str | None = None
    dealer: str | None = None
    seat: str | None = None
    interestingSeats: list[str]
    relatedSakkas: list[SakkaReferenceV1]
    renderCount: int = Field(ge=0)
    renderStatus: Literal[
        "not_prepared", "prepared", "queued", "running", "completed", "failed", "cancelled"
    ]
    createdAt: str
    traceHash: str | None = None
    gameHash: str | None = None
    sourceLocator: LaffaSourceReferenceV1
    sourceReferences: list[LaffaSourceReferenceV1]


class LaffaSourceFacetV1(BaseModel):
    sourceKind: Literal[
        "archived_game",
        "sakka_round",
        "research_result",
        "qualified_laffa",
        "research_audit_trace",
    ]
    count: int = Field(ge=0)


class UnifiedLaffaLibraryResponseV1(BaseModel):
    schemaVersion: Literal[1]
    resourceKind: Literal["unified_laffa_library_v1"]
    items: list[UnifiedLaffaRowV1]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    nextOffset: int | None = Field(default=None, ge=0)
    sourceFacets: list[LaffaSourceFacetV1]


def _page(values: list[dict[str, Any]], *, limit: int, cursor_key: str) -> dict[str, Any]:
    size = max(1, min(500, int(limit)))
    selected = values[:size]
    return {
        "items": selected,
        "nextCursor": selected[-1][cursor_key] if len(values) > size else None,
    }


def build_foundry_admin_v2_router(
    *,
    campaign_control: Any,
    learning_store: Any,
    learning: Any,
    hitl: Any,
    configuration: Any,
    registry: Any,
    foundry_catalogs: Any,
    context_searches: Any,
    sakka_generations: Any,
    identities: Any,
    require_trusted: Callable[[Request], None],
    publication_policy: Mapping[str, Any],
    research: Any | None = None,
    render_batches: Any | None = None,
    active_bundle_hash: str | None = None,
    laffa_library: LaffaLibraryServiceV1 | None = None,
) -> APIRouter:
    """Expose resource-oriented V2 adapters without duplicating domain logic."""

    router = APIRouter(prefix="/api/v2", tags=["foundry-admin-v2"])
    laffa_library = laffa_library or LaffaLibraryServiceV1(
        learning_store=learning_store,
        research=research,
        foundry_catalogs=foundry_catalogs,
        render_batches=render_batches,
        active_bundle_hash=active_bundle_hash,
    )

    @router.post("/sakka-generations", status_code=202)
    def create_sakka_generation(body: NaturalSakkaGenerationCreateV2, request: Request):
        require_trusted(request)
        require_identity(request, identities)
        idempotency_key = request.headers.get("idempotency-key", "").strip()
        if not idempotency_key:
            raise HTTPException(428, "Idempotency-Key header is required")
        try:
            return sakka_generations.submit(
                root_seed=body.rootSeed,
                initial_dealer=body.initialDealer,
                idempotency_key=idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/sakka-generations/{job_id}")
    def get_sakka_generation(job_id: str, request: Request):
        require_trusted(request)
        item = sakka_generations.get(job_id)
        if item is None:
            raise HTTPException(404, "natural Sakka generation job not found")
        return item

    @router.post("/sakka-generations/{job_id}:cancel")
    def cancel_sakka_generation(job_id: str, request: Request):
        require_trusted(request)
        item = sakka_generations.get(job_id)
        if item is None:
            raise HTTPException(404, "natural Sakka generation job not found")
        if not sakka_generations.cancel(job_id):
            raise HTTPException(409, f"natural Sakka generation cannot be cancelled from {item['status']}")
        return sakka_generations.get(job_id)

    @router.get("/sakka-generations/{job_id}/events")
    def stream_sakka_generation_events(job_id: str, request: Request):
        require_trusted(request)
        if sakka_generations.get(job_id) is None:
            raise HTTPException(404, "natural Sakka generation job not found")

        def stream():
            seen = 0
            while True:
                events = sakka_generations.events(job_id) or []
                for index, event in enumerate(events[seen:], start=seen):
                    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {index + 1}\nevent: {event['type']}\ndata: {encoded}\n\n"
                seen = len(events)
                item = sakka_generations.get(job_id)
                if item is None or item["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/context-searches", status_code=202)
    def create_context_search(body: ContextSearchCreateV2, request: Request):
        require_trusted(request)
        require_identity(request, identities)
        idempotency_key = request.headers.get("idempotency-key", "").strip()
        if not idempotency_key:
            raise HTTPException(428, "Idempotency-Key header is required")
        try:
            return context_searches.submit(
                parent_trace_reference=body.parentTraceReference.model_dump(),
                decision_id=body.decisionId,
                context_proof_hashes=body.contextProofHashes,
                idempotency_key=idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/context-searches/{job_id}")
    def get_context_search(job_id: str, request: Request):
        require_trusted(request)
        item = context_searches.get(job_id)
        if item is None:
            raise HTTPException(404, "context search job not found")
        return item

    @router.post("/context-searches/{job_id}:cancel")
    def cancel_context_search(job_id: str, request: Request):
        require_trusted(request)
        item = context_searches.get(job_id)
        if item is None:
            raise HTTPException(404, "context search job not found")
        if not context_searches.cancel(job_id):
            raise HTTPException(409, f"context search cannot be cancelled from {item['status']}")
        return context_searches.get(job_id)

    @router.get("/context-searches/{job_id}/events")
    def stream_context_search_events(job_id: str, request: Request):
        require_trusted(request)
        if context_searches.get(job_id) is None:
            raise HTTPException(404, "context search job not found")

        def stream():
            seen = 0
            while True:
                events = context_searches.events(job_id) or []
                for index, event in enumerate(events[seen:], start=seen):
                    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    yield f"id: {index + 1}\nevent: {event['type']}\ndata: {encoded}\n\n"
                seen = len(events)
                item = context_searches.get(job_id)
                if item is None or item["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/shards")
    def list_shards(
        request: Request,
        campaignHash: str,
        after: str | None = None,
        limit: int = 100,
    ):
        require_trusted(request)
        if campaign_control.get_campaign(campaignHash) is None:
            raise HTTPException(404, "Foundry campaign not found")
        values = campaign_control.list_shards(campaignHash)
        if after:
            values = [item for item in values if str(item["specHash"]) > after]
        values.sort(key=lambda item: str(item["specHash"]))
        return _page(values, limit=limit, cursor_key="specHash")

    @router.get("/laffat", response_model=UnifiedLaffaLibraryResponseV1)
    def list_laffat(
        request: Request,
        search: str = Query(default="", max_length=500),
        sourceKind: Literal[
            "archived_game",
            "sakka_round",
            "research_result",
            "qualified_laffa",
            "research_audit_trace",
        ] | None = None,
        replayValid: bool | None = None,
        qualificationStatus: str | None = Query(default=None, max_length=100),
        truthLabel: str | None = Query(default=None, max_length=100),
        contract: str | None = Query(default=None, max_length=100),
        project: str | None = Query(default=None, max_length=100),
        tactic: str | None = Query(default=None, max_length=100),
        seat: Literal["north", "east", "south", "west"] | None = None,
        renderStatus: Literal[
            "not_prepared", "prepared", "queued", "running", "completed", "failed", "cancelled"
        ] | None = None,
        sort: Literal["newest", "oldest", "score_desc", "confidence_desc", "code_asc"] = "newest",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ):
        require_trusted(request)
        try:
            return laffa_library.list(
                search=search,
                source_kind=sourceKind,
                replay_valid=replayValid,
                qualification_status=qualificationStatus,
                truth_label=truthLabel,
                contract=contract,
                project=project,
                tactic=tactic,
                seat=seat,
                render_status=renderStatus,
                sort=sort,
                offset=offset,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/laffat/{identifier}", response_model=UnifiedLaffaRowV1)
    def get_laffa(identifier: str, request: Request):
        require_trusted(request)
        try:
            row = laffa_library.get(identifier)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if row is None:
            raise HTTPException(404, "laffa not found")
        return row

    @router.get("/sakkas")
    def list_sakkas(
        request: Request,
        provenance: Literal["natural", "curated"] | None = None,
        after: str | None = None,
        limit: int = 100,
    ):
        require_trusted(request)
        try:
            return foundry_catalogs.list_sakkas(
                provenance=provenance, after=after, limit=limit
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/sakkas/natural", status_code=201)
    def create_natural_sakka(body: NaturalSakkaCreateV2, request: Request):
        require_trusted(request)
        require_identity(request, identities)
        try:
            return foundry_catalogs.create_natural_sakka(
                tuple(item.model_dump() for item in body.traceReferences),
                target=body.target,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(404, "complete-trace artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/sakkas/curated", status_code=201)
    def create_curated_sakka(body: CuratedSakkaCreateV2, request: Request):
        require_trusted(request)
        require_identity(request, identities)
        try:
            return foundry_catalogs.create_curated_sakka(
                tuple(item.model_dump() for item in body.selections),
                target=body.target,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(404, "contextualized Laffa artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/sakkas/{sakka_hash}")
    def get_sakka(sakka_hash: str, request: Request):
        require_trusted(request)
        try:
            foundry_catalogs.load_sakka(sakka_hash)
            return foundry_catalogs.get(sakka_hash)
        except KeyError as exc:
            raise HTTPException(404, "Foundry Sakka not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/config-revisions")
    def list_config_revisions(
        request: Request,
        kind: Literal["profile", "bundle"] = "profile",
        status: str | None = None,
        limit: int = 100,
    ):
        require_trusted(request)
        size = max(1, min(500, int(limit)))
        if kind == "profile":
            values = configuration.store.list_profiles(status=status, limit=size)
            items = [item.to_dict(include_payload=False) for item in values]
        else:
            values = configuration.store.list_bundles(status=status, limit=size)
            items = [item.to_dict() for item in values]
        return {"kind": kind, "items": items}

    @router.get("/plugins")
    def list_plugins(request: Request, kind: str | None = None):
        require_trusted(request)
        return {
            "items": registry.list(kind),
            "externalInProcessAllowed": False,
            "policy": registry.policy.to_dict(),
        }

    @router.get("/corrections")
    def list_corrections(request: Request, limit: int = 100):
        require_trusted(request)
        return {"items": hitl.list_cases(limit=max(1, min(500, int(limit))))}

    @router.post("/corrections", status_code=201)
    def create_correction(body: CorrectionCreateV2, request: Request):
        require_trusted(request)
        identity, _session = require_identity(request, identities)
        try:
            return hitl.create_case(
                body.decisionId,
                created_by=identity.subject_id,
                correction_type=body.correctionType,
                high_impact=body.highImpact,
                completed=body.completed,
            )
        except KeyError as exc:
            raise HTTPException(404, "decision not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/datasets")
    def list_datasets(request: Request):
        require_trusted(request)
        return {"items": learning_store.list_datasets()}

    @router.post("/datasets", status_code=201)
    def create_dataset(body: DatasetCreateV2, request: Request):
        require_trusted(request)
        require_identity(request, identities)
        try:
            return learning.build_dataset(
                purpose=body.purpose,
                minimum_confidence=body.minimumConfidence,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/models")
    def list_models(request: Request):
        require_trusted(request)
        return {"items": learning_store.list_models()}

    @router.post("/models", status_code=202)
    def create_model(body: ModelCreateV2, request: Request):
        require_trusted(request)
        require_identity(request, identities)
        overrides = {
            key: value
            for key, value in {
                "epochs": body.epochs,
                "learning_rate": body.learningRate,
                "l2": body.l2,
                "batch_size": body.batchSize,
                "seed": body.seed,
            }.items()
            if value is not None
        }
        try:
            return learning.train_model(
                body.datasetId,
                overrides=overrides,
                backend=body.backend,
            )
        except KeyError as exc:
            raise HTTPException(404, "dataset not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/publications")
    def list_publications(
        request: Request,
        status: Literal["review", "approved", "published", "retired"] | None = None,
        limit: int = 100,
    ):
        require_trusted(request)
        selected = status or "published"
        return {
            "items": learning_store.list_challenges(
                status=selected,
                limit=max(1, min(500, int(limit))),
            ),
            "status": selected,
            "automaticPublishing": False,
        }

    @router.post("/publications")
    def transition_publication(body: PublicationTransitionV2, request: Request):
        require_trusted(request)
        identity, _session = require_identity(request, identities)
        actor_id, role = identity.subject_id, identity.kind
        try:
            if body.action in {"review", "reject"}:
                validate_reviewer_role(role, publication_policy)
                target = "review" if body.action == "review" else "rejected"
                return learning_store.update_challenge_status(body.challengeId, target)
            target = {
                "approve": "approved",
                "publish": "published",
                "retire": "retired",
            }[body.action]
            if body.evidenceSha256 is None:
                raise ValueError("governed publication transition requires evidenceSha256")
            return learning_store.govern_challenge_status(
                body.challengeId,
                target=target,
                actor_id=actor_id,
                actor_role=role,
                evidence_sha256=body.evidenceSha256,
                policy=publication_policy,
            )
        except KeyError as exc:
            raise HTTPException(404, "challenge not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
