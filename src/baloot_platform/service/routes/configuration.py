from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from baloot_platform.configuration import VersionedConfigurationService
from baloot_platform.identity import IdentityService
from baloot_platform.security import AuthorizationPolicy, Scope

from .auth import require_identity


class ProfileCreateRequest(BaseModel):
    family: str
    payload: dict[str, Any]
    scope: str = "global"
    scopeKey: str | None = None
    tenantId: str | None = None
    environment: str | None = None
    effectiveFrom: str | None = None
    effectiveUntil: str | None = None
    rolloutPercentage: float = Field(100.0, gt=0, le=100)
    rolloutSeed: str = "default"
    supersedesRevisionId: str | None = None
    publish: bool = False


class PublishRequest(BaseModel):
    expectedEtag: int = Field(ge=1)


class ProfilePatchRequest(BaseModel):
    payload: dict[str, Any]
    expectedEtag: int = Field(ge=1)


class TransitionRequest(BaseModel):
    expectedEtag: int = Field(ge=1)
    reason: str = Field(min_length=2, max_length=2000)


class BundleCreateRequest(BaseModel):
    bundleId: str = Field(min_length=2, max_length=160)
    version: str
    mode: str = "full"
    componentRevisions: dict[str, str]
    scope: str = "global"
    scopeKey: str | None = None
    tenantId: str | None = None
    environment: str | None = None
    effectiveFrom: str | None = None
    effectiveUntil: str | None = None
    rolloutPercentage: float = Field(100.0, gt=0, le=100)
    rolloutSeed: str = "default"
    supersedesRevisionId: str | None = None
    publish: bool = False
    allowDraftComponents: bool = False


class BootstrapRequest(BaseModel):
    mode: str = "full"


class RollbackRequest(BaseModel):
    newVersion: str


class ConfigurationBundleRevisionResponse(BaseModel):
    model_config = {"extra": "forbid"}

    revisionId: str
    bundleId: str
    version: str
    mode: str
    scope: str
    scopeType: str
    scopeKey: str
    tenantId: str | None
    environment: str | None
    status: str
    componentRevisions: dict[str, str]
    contentSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=1)
    effectiveFrom: str | None
    effectiveUntil: str | None
    rolloutPercentage: float = Field(gt=0.0, le=100.0)
    rolloutSeed: str
    createdBy: str
    createdAt: str
    updatedAt: str
    publishedBy: str | None
    publishedAt: str | None
    supersedesRevisionId: str | None
    etag: int = Field(ge=1)


class ConfigurationBundleListResponse(BaseModel):
    model_config = {"extra": "forbid"}

    items: list[ConfigurationBundleRevisionResponse]


def _role(request: Request) -> str:
    claims = getattr(request.state, "access_claims", None)
    if claims is not None:
        return str(getattr(claims, "role", "player")).lower()
    if bool(getattr(request.state, "api_key_valid", False)):
        return request.headers.get("x-baloot-role", "service").lower()
    return "player"


def _admin(request: Request, policy: AuthorizationPolicy) -> None:
    claims = getattr(request.state, "access_claims", None)
    try:
        policy.require(role=_role(request), required=Scope.CONFIG_ADMIN, claims=claims)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


def build_configuration_router(service: VersionedConfigurationService, identities: IdentityService,
                               policy: AuthorizationPolicy, profile: dict[str, Any],
                               performance_profile: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/api/v1/configuration-registry", tags=["configuration-control-plane"])

    @router.get("/profile")
    def registry_profile(request: Request):
        require_identity(request, identities)
        return {
            "profile": profile,
            "performanceProfile": performance_profile,
            "automaticProductionApply": False,
            "authoritativeRuntimeRequiresPublishedBundle": True,
        }

    @router.get("/profiles")
    def list_profiles(request: Request, family: str | None = None, profileId: str | None = None,
                      scope: str | None = None, scopeKey: str | None = None, tenantId: str | None = None,
                      environment: str | None = None, status: str | None = None, limit: int = 200):
        require_identity(request, identities)
        values = service.store.list_profiles(
            family=family, profile_id=profileId, scope=scope, scope_key=scopeKey, tenant_id=tenantId,
            environment=environment, status=status, limit=limit,
        )
        return {"items": [item.to_dict(include_payload=False) for item in values]}

    @router.get("/profile-schemas/{family}")
    def get_profile_schema(family: str, request: Request):
        require_identity(request, identities)
        try:
            return service.profile_schema(family)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.get("/profiles:resolve")
    def resolve_profile(request: Request, family: str, profileId: str, tenantId: str | None = None,
                        environment: str | None = None, roomId: str | None = None,
                        sharedGameId: str | None = None, videoProfileId: str | None = None,
                        experimentKey: str | None = None, subjectId: str | None = None, atTime: str | None = None):
        require_identity(request, identities)
        try:
            return service.resolve_profile(family=family, profile_id=profileId, tenant_id=tenantId,
                environment=environment, room_id=roomId, shared_game_id=sharedGameId,
                video_profile_id=videoProfileId, user_experiment_key=experimentKey,
                subject_id=subjectId, at_time=atTime)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/profiles:compare")
    def compare_profiles(request: Request, left: str, right: str):
        require_identity(request, identities)
        left_value = service.store.get_profile(left)
        right_value = service.store.get_profile(right)
        if left_value is None or right_value is None:
            raise HTTPException(404, "configuration profile revision not found")
        return {
            "left": left_value.to_dict(include_payload=False),
            "right": right_value.to_dict(include_payload=False),
            "changes": service.compare_payloads(dict(left_value.payload), dict(right_value.payload)),
        }

    @router.post("/profiles", status_code=201)
    def create_profile(body: ProfileCreateRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.create_profile(
                family=body.family, payload=body.payload, created_by=identity.subject_id,
                scope=body.scope, scope_key=body.scopeKey, tenant_id=body.tenantId,
                environment=body.environment, effective_from=body.effectiveFrom, effective_until=body.effectiveUntil,
                rollout_percentage=body.rolloutPercentage, rollout_seed=body.rolloutSeed,
                supersedes_revision_id=body.supersedesRevisionId, publish=body.publish,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/profiles/{revision_id}")
    def get_profile(revision_id: str, request: Request, includePayload: bool = True):
        require_identity(request, identities)
        value = service.store.get_profile(revision_id)
        if value is None:
            raise HTTPException(404, "configuration profile revision not found")
        return value.to_dict(include_payload=includePayload)

    @router.patch("/profiles/{revision_id}")
    def update_profile(
        revision_id: str, body: ProfilePatchRequest, request: Request
    ):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.update_profile(
                revision_id,
                payload=body.payload,
                expected_etag=body.expectedEtag,
                actor_id=identity.subject_id,
            )
        except KeyError as exc:
            raise HTTPException(404, "configuration profile revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/profiles/{revision_id}:validate")
    def validate_profile(revision_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.validate_profile_revision(revision_id)
        except KeyError as exc:
            raise HTTPException(404, "configuration profile revision not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/profiles/{revision_id}:publish")
    def publish_profile(revision_id: str, body: PublishRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.publish_profile(revision_id, expected_etag=body.expectedEtag, actor_id=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "configuration profile revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/profiles/{revision_id}:reject")
    def reject_profile(revision_id: str, body: TransitionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.reject_profile(revision_id, expected_etag=body.expectedEtag, actor_id=identity.subject_id, reason=body.reason)
        except KeyError as exc:
            raise HTTPException(404, "configuration profile revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/profiles/{revision_id}:retire")
    def retire_profile(revision_id: str, body: TransitionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.retire_profile(revision_id, expected_etag=body.expectedEtag, actor_id=identity.subject_id, reason=body.reason)
        except KeyError as exc:
            raise HTTPException(404, "configuration profile revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/profiles/{revision_id}/events")
    def profile_events(revision_id: str, request: Request):
        require_identity(request, identities)
        if service.store.get_profile(revision_id) is None:
            raise HTTPException(404, "configuration profile revision not found")
        return service.store.events_with_verification("profile", revision_id)

    @router.get("/bundles", response_model=ConfigurationBundleListResponse)
    def list_bundles(request: Request, bundleId: str | None = None, scope: str | None = None,
                     scopeKey: str | None = None, tenantId: str | None = None,
                     environment: str | None = None, status: str | None = None, limit: int = 200):
        require_identity(request, identities)
        values = service.store.list_bundles(
            bundle_id=bundleId, scope=scope, scope_key=scopeKey, tenant_id=tenantId, environment=environment, status=status, limit=limit,
        )
        return {"items": [item.to_dict() for item in values]}

    @router.post("/bundles", status_code=201)
    def create_bundle(body: BundleCreateRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.create_bundle(
                bundle_id=body.bundleId, version=body.version, mode=body.mode,
                component_revisions=body.componentRevisions, created_by=identity.subject_id,
                scope=body.scope, scope_key=body.scopeKey, tenant_id=body.tenantId,
                environment=body.environment, effective_from=body.effectiveFrom, effective_until=body.effectiveUntil,
                rollout_percentage=body.rolloutPercentage, rollout_seed=body.rolloutSeed,
                supersedes_revision_id=body.supersedesRevisionId, publish=body.publish,
                allow_draft_components=body.allowDraftComponents,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/bundles/{revision_id}")
    def get_bundle(revision_id: str, request: Request):
        require_identity(request, identities)
        value = service.store.get_bundle(revision_id)
        if value is None:
            raise HTTPException(404, "configuration bundle revision not found")
        return value.to_dict()

    @router.post("/bundles/{revision_id}:compile")
    def compile_bundle(revision_id: str, request: Request):
        require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.compile_bundle(revision_id)
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/bundles/{revision_id}:publish")
    def publish_bundle(revision_id: str, body: PublishRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.publish_bundle(revision_id, expected_etag=body.expectedEtag, actor_id=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/bundles/{revision_id}:resolve")
    def resolve_bundle(revision_id: str, request: Request, allowDraft: bool = False):
        require_identity(request, identities)
        if allowDraft:
            _admin(request, policy)
        try:
            return service.resolve(revision_id=revision_id, allow_draft=allowDraft)
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/bundles/{revision_id}:resolve-v2")
    def resolve_bundle_v2(revision_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.freeze_v2(revision_id=revision_id)
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/bundles/{revision_id}:rollback", status_code=201)
    def rollback_bundle(
        revision_id: str, body: RollbackRequest, request: Request
    ):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.rollback_bundle(
                revision_id,
                new_version=body.newVersion,
                actor_id=identity.subject_id,
            )
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/bundles/{revision_id}:reject")
    def reject_bundle(revision_id: str, body: TransitionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.reject_bundle(revision_id, expected_etag=body.expectedEtag, actor_id=identity.subject_id, reason=body.reason)
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/bundles/{revision_id}:retire")
    def retire_bundle(revision_id: str, body: TransitionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.retire_bundle(revision_id, expected_etag=body.expectedEtag, actor_id=identity.subject_id, reason=body.reason)
        except KeyError as exc:
            raise HTTPException(404, "configuration bundle revision not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/bundles/{revision_id}/events")
    def bundle_events(revision_id: str, request: Request):
        require_identity(request, identities)
        if service.store.get_bundle(revision_id) is None:
            raise HTTPException(404, "configuration bundle revision not found")
        return service.store.events_with_verification("bundle", revision_id)

    @router.get("/resolve")
    def resolve_current(request: Request, bundleId: str | None = None, tenantId: str | None = None,
                        environment: str | None = None, roomId: str | None = None,
                        sharedGameId: str | None = None, videoProfileId: str | None = None,
                        experimentKey: str | None = None, subjectId: str | None = None, atTime: str | None = None):
        require_identity(request, identities)
        try:
            return service.resolve(bundle_id=bundleId, tenant_id=tenantId, environment=environment,
                room_id=roomId, shared_game_id=sharedGameId, video_profile_id=videoProfileId,
                user_experiment_key=experimentKey, subject_id=subjectId, at_time=atTime)
        except KeyError as exc:
            raise HTTPException(404, "published configuration bundle not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/resolve-v2")
    def resolve_current_v2(
        request: Request,
        bundleId: str | None = None,
        tenantId: str | None = None,
        environment: str | None = None,
        roomId: str | None = None,
        sharedGameId: str | None = None,
        videoProfileId: str | None = None,
        experimentKey: str | None = None,
        subjectId: str | None = None,
        atTime: str | None = None,
    ):
        require_identity(request, identities)
        try:
            return service.freeze_v2(
                bundle_id=bundleId,
                tenant_id=tenantId,
                environment=environment,
                room_id=roomId,
                shared_game_id=sharedGameId,
                video_profile_id=videoProfileId,
                user_experiment_key=experimentKey,
                subject_id=subjectId,
                at_time=atTime,
            )
        except KeyError as exc:
            raise HTTPException(404, "published configuration bundle not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/bootstrap")
    @router.post(":bootstrap", include_in_schema=False)
    def bootstrap(body: BootstrapRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _admin(request, policy)
        try:
            return service.bootstrap_packaged_bundle(actor_id=identity.subject_id, mode=body.mode)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    aliases = APIRouter(prefix="/api/v2/configuration", tags=["configuration-control-plane-v2"])
    aliases.add_api_route("/profiles", list_profiles, methods=["GET"], name="v2_list_configuration_profiles")
    aliases.add_api_route("/profile-schemas/{family}", get_profile_schema, methods=["GET"], name="v2_get_configuration_profile_schema")
    aliases.add_api_route("/profiles", create_profile, methods=["POST"], status_code=201, name="v2_create_configuration_profile")
    aliases.add_api_route("/profiles/{revision_id}", get_profile, methods=["GET"], name="v2_get_configuration_profile")
    aliases.add_api_route("/profiles/{revision_id}", update_profile, methods=["PATCH"], name="v2_update_configuration_profile")
    aliases.add_api_route("/profiles/{revision_id}:validate", validate_profile, methods=["POST"], name="v2_validate_configuration_profile")
    aliases.add_api_route("/profiles/{revision_id}:publish", publish_profile, methods=["POST"], name="v2_publish_configuration_profile")
    aliases.add_api_route("/profiles/{revision_id}:retire", retire_profile, methods=["POST"], name="v2_retire_configuration_profile")
    aliases.add_api_route(
        "/bundles",
        list_bundles,
        methods=["GET"],
        name="v2_list_configuration_bundles",
        response_model=ConfigurationBundleListResponse,
    )
    aliases.add_api_route("/bundles", create_bundle, methods=["POST"], status_code=201, name="v2_create_configuration_bundle")
    aliases.add_api_route("/bundles/{revision_id}", get_bundle, methods=["GET"], name="v2_get_configuration_bundle")
    aliases.add_api_route("/bundles/{revision_id}:compile", compile_bundle, methods=["POST"], name="v2_compile_configuration_bundle")
    aliases.add_api_route("/bundles/{revision_id}:publish", publish_bundle, methods=["POST"], name="v2_publish_configuration_bundle")
    aliases.add_api_route("/bundles/{revision_id}:retire", retire_bundle, methods=["POST"], name="v2_retire_configuration_bundle")

    outer = APIRouter()
    outer.include_router(router)
    outer.include_router(aliases)
    return outer
