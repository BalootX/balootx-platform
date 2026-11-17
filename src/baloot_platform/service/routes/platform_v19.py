from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from baloot_platform.benchmark import IntelligenceBenchmarkV18
from baloot_platform.certification import verify_certification_candidate
from baloot_platform.configuration import VersionedConfigurationService
from baloot_platform.identity import IdentityService
from baloot_platform.performance import PerformanceService
from baloot_platform.plugins.registry import PluginRegistry
from baloot_platform.security import AuthorizationPolicy, Scope

from .auth import require_identity


class CertificationVerifyRequest(BaseModel):
    candidate: dict[str, Any]


class BenchmarkEvaluateRequest(BaseModel):
    modelId: str = Field(min_length=1, max_length=200)
    examples: list[dict[str, Any]] = Field(min_length=1, max_length=100000)


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


def build_platform_v19_router(*, identities: IdentityService, policy: AuthorizationPolicy,
                              registry: PluginRegistry, performance: PerformanceService,
                              configuration: VersionedConfigurationService,
                              capability_matrix: dict[str, Any],
                              certification_profile: dict[str, Any],
                              benchmark_profile: dict[str, Any],
                              operations_profile: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["platform-v20"])
    benchmark = IntelligenceBenchmarkV18(benchmark_profile)

    @router.get("/platform/capabilities")
    def capabilities(request: Request):
        return capability_matrix

    @router.get("/plugins")
    def plugins(request: Request, kind: str | None = None):
        items = registry.list(kind)
        return {"items": items, "plugins": items, "externalInProcessAllowed": False,
                "policy": registry.policy.to_dict()}

    @router.get("/performance/profile")
    def performance_profile(request: Request):
        return performance.profile

    @router.post("/performance/quick")
    def performance_quick(request: Request):
        require_identity(request, identities)
        _admin(request, policy)
        measurements = [
            performance.measure("configuration_resolve_p95_ms", configuration.resolve, iterations=50, warmup=5),
            performance.measure("plugin_registry_list_p95_ms", lambda: registry.list(), iterations=100, warmup=10),
            performance.measure("canonical_payload_p95_ms", lambda: json.dumps(capability_matrix, ensure_ascii=False, sort_keys=True, separators=(",", ":")), iterations=100, warmup=10),
        ]
        return performance.report(measurements)

    @router.get("/certification/profile")
    def certification(request: Request):
        return {"profile": certification_profile, "automaticRuleActivation": False}

    @router.post("/certification/candidates:verify")
    def certification_verify(body: CertificationVerifyRequest, request: Request):
        require_identity(request, identities)
        _admin(request, policy)
        return verify_certification_candidate(
            body.candidate,
            trusted_signers=certification_profile.get("trusted_signers", ()),
            required_signer_roles=certification_profile["required_signer_roles"],
            minimum_external_signatures=certification_profile[
                "minimum_external_signatures"
            ],
            expected_profile_identity=certification_profile.get(
                "candidate_profile_identity"
            ),
            expected_source_catalog_identity=certification_profile.get(
                "rule_source_catalog"
            ),
            expected_adjudication_profile_hash=(
                certification_profile.get("adjudication_authority_identity")
                or {}
            ).get("profile_sha256"),
            certification_eligible=bool(
                certification_profile.get("certification_eligible", True)
            ),
        )

    @router.get("/benchmark/profile")
    def benchmark_profile_route(request: Request):
        return {"profile": benchmark_profile, "automaticPromotion": False}

    @router.post("/benchmark/evaluations")
    def benchmark_evaluate(body: BenchmarkEvaluateRequest, request: Request):
        require_identity(request, identities)
        _admin(request, policy)
        try:
            return benchmark.evaluate(body.modelId, body.examples).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/operations/profile")
    def operations(request: Request):
        return {"profile": operations_profile, "automaticProductionChange": False}

    return router
