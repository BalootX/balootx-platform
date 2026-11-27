from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from baloot_platform.identity import Identity, IdentityService
from baloot_platform.identity.models import DISPLAY_NAME_MAX_LENGTH
from baloot_platform.identity.pagination import (
    CURSOR_MAX_LENGTH,
    IdentityPaginationConfig,
    InvalidIdentityCursor,
)
from baloot_platform.security import AuthorizationPolicy, Scope
from baloot_platform.service.authorization import claims_identity
from baloot_platform.service.client_identity import service_client_identity

COOKIE_NAME = "baloot_session"


class GuestRequest(BaseModel):
    displayName: str | None = Field(default=None, max_length=DISPLAY_NAME_MAX_LENGTH)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)
    displayName: str | None = Field(default=None, max_length=DISPLAY_NAME_MAX_LENGTH)


class LoginRequest(BaseModel):
    username: str
    password: str


class FriendRequestBody(BaseModel):
    recipientId: str


class FriendResponseBody(BaseModel):
    accept: bool = True


class IdentityItem(BaseModel):
    subjectId: str
    kind: str
    displayName: str
    username: str | None
    authenticated: bool
    guest: bool
    createdAt: str | None


class PendingRequestItem(BaseModel):
    requestId: str = Field(validation_alias="request_id")
    senderId: str = Field(validation_alias="sender_id")
    recipientId: str = Field(validation_alias="recipient_id")
    status: Literal["pending"]
    createdAt: str = Field(validation_alias="created_at")
    senderDisplayName: str = Field(validation_alias="sender_display_name")
    senderUsername: str | None = Field(validation_alias="sender_username")


class IdentityPage(BaseModel):
    items: list[IdentityItem]
    nextCursor: str | None


class PendingRequestPage(BaseModel):
    items: list[PendingRequestItem]
    nextCursor: str | None


def request_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.cookies.get(COOKIE_NAME)


def current_identity(request: Request, service: IdentityService) -> tuple[Identity | None, dict[str, Any] | None]:
    identity, session = service.resolve(request_token(request))
    if identity is not None and session is not None:
        return identity, session
    claims = getattr(request.state, "access_claims", None)
    if claims is not None:
        synthetic = claims_identity(
            claims, request.headers.get("x-baloot-subject-name", claims.subject_id),
        )
        return synthetic, {
            "session_id": getattr(claims, "session_id", None),
            "csrf_token": "",
            "expires_at": getattr(claims, "expires_at", None),
            "access_claims": claims,
        }
    return None, None


def require_identity(request: Request, service: IdentityService) -> tuple[Identity, dict[str, Any]]:
    identity, session = current_identity(request, service)
    if identity is not None and session is not None:
        return identity, session
    if bool(getattr(request.state, "api_key_valid", False)):
        return service_client_identity(request.headers, "API Client"), {"csrf_token": "", "service_key": True}
    raise HTTPException(401, "ابدأ كضيف أو سجّل الدخول")


def _cookie_secure(request: Request) -> bool:
    environment = os.environ.get("BALOOT_ENV", "development").lower()
    return environment not in {"development", "test"} and request.url.scheme == "https"


def apply_session_cookie(response: Response, request: Request, result: dict[str, Any]) -> None:
    session = result["session"]
    response.set_cookie(
        COOKIE_NAME,
        session["token"],
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


def build_auth_router(
    service: IdentityService, *, pagination: IdentityPaginationConfig, policy: AuthorizationPolicy
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["identity"])
    paged_router = APIRouter(prefix="/api/v2", tags=["identity"])

    def actor(request: Request, scope: Scope, *, require_signed_scope: bool = False) -> Identity:
        identity, session = require_identity(request, service)
        claims = session.get("access_claims")
        if session.get("service_key") or (require_signed_scope and claims is not None):
            try:
                policy.require(role=identity.kind, required=scope, claims=claims)
            except PermissionError as exc:
                raise HTTPException(403, str(exc)) from exc
        return identity

    def page(fetch: Callable[..., dict[str, Any]], value: str, limit: int | None, cursor: str | None):
        page_size = min(limit, pagination.maximum_page_size) if limit is not None else pagination.default_page_size
        try:
            return fetch(value, limit=page_size, cursor=cursor)
        except InvalidIdentityCursor as exc:
            raise HTTPException(422, [{"loc": ["query", "cursor"], "msg": str(exc), "type": "value_error"}]) from exc

    @router.get("/auth/me")
    def me(request: Request):
        identity, session = current_identity(request, service)
        if identity is None or session is None:
            return {"identity": None, "signInOptional": True}
        return {
            "identity": identity.to_dict(),
            "csrfToken": session["csrf_token"],
            "expiresAt": session["expires_at"],
            "signInOptional": True,
        }

    @router.post("/auth/guest", status_code=201)
    def guest(body: GuestRequest, request: Request, response: Response):
        identity, session = current_identity(request, service)
        if identity is not None and session is not None:
            return {"identity": identity.to_dict(), "csrfToken": session["csrf_token"], "reused": True}
        result = service.create_guest(body.displayName, user_agent=request.headers.get("user-agent"))
        apply_session_cookie(response, request, result)
        return {
            "identity": result["identity"], "csrfToken": result["session"]["csrfToken"],
            "accessToken": result["session"]["token"], "reused": False,
        }

    @router.post("/auth/register", status_code=201)
    def register(body: RegisterRequest, request: Request, response: Response):
        current, _current_session = current_identity(request, service)
        try:
            result = service.register(
                body.username, body.password, body.displayName,
                user_agent=request.headers.get("user-agent"),
                upgrade_guest_id=current.subject_id if current is not None and current.is_guest else None,
            )
        except ValueError as exc:
            raise HTTPException(409 if "exists" in str(exc) else 422, str(exc)) from exc
        apply_session_cookie(response, request, result)
        return {
            "identity": result["identity"], "csrfToken": result["session"]["csrfToken"],
            "accessToken": result["session"]["token"],
        }

    @router.post("/auth/login")
    def login(body: LoginRequest, request: Request, response: Response):
        try:
            result = service.login(body.username, body.password, user_agent=request.headers.get("user-agent"))
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        apply_session_cookie(response, request, result)
        return {
            "identity": result["identity"], "csrfToken": result["session"]["csrfToken"],
            "accessToken": result["session"]["token"],
        }

    @router.post("/auth/logout", status_code=204)
    def logout(request: Request):
        service.logout(request_token(request))
        response = Response(status_code=204)
        response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=_cookie_secure(request))
        return response

    @router.get("/users:search")
    def search_users(request: Request, query: str, limit: int = 20):
        require_identity(request, service)
        return {"items": service.search(query, limit=limit)}

    @paged_router.get("/users:search", response_model=IdentityPage)
    def search_users_page(
        request: Request, query: str, limit: int | None = Query(default=None, ge=1),
        cursor: str | None = Query(default=None, max_length=CURSOR_MAX_LENGTH),
    ):
        identity = actor(request, Scope.READ, require_signed_scope=True)
        if identity.is_guest and not pagination.allow_guest_search:
            raise HTTPException(403, "sign in to search for users")
        maximum = pagination.maximum_search_query_length
        if len(query) > maximum:
            raise HTTPException(422, [{
                "loc": ["query", "query"],
                "msg": f"search query must contain at most {maximum} characters",
                "type": "value_error",
            }])
        minimum = pagination.minimum_search_query_length
        if "\0" in query or len(query.strip()) < minimum:
            raise HTTPException(422, [{
                "loc": ["query", "query"],
                "msg": f"search query must contain at least {minimum} characters and no nulls",
                "type": "value_error",
            }])
        return page(partial(service.search_page, maximum_results=pagination.maximum_search_results), query, limit, cursor)

    @router.get("/friends")
    def friends(request: Request):
        identity = actor(request, Scope.READ)
        return {"items": service.friends(identity.subject_id)}

    @router.get("/friends/requests")
    def friend_requests(request: Request):
        identity = actor(request, Scope.READ)
        return {"items": service.requests(identity.subject_id)}

    @paged_router.get("/friends", response_model=IdentityPage)
    def friends_page(
        request: Request, limit: int | None = Query(default=None, ge=1),
        cursor: str | None = Query(default=None, max_length=CURSOR_MAX_LENGTH),
    ):
        identity = actor(request, Scope.READ, require_signed_scope=True)
        return page(service.friend_page, identity.subject_id, limit, cursor)

    @paged_router.get("/friends/requests", response_model=PendingRequestPage)
    def friend_requests_page(
        request: Request, limit: int | None = Query(default=None, ge=1),
        cursor: str | None = Query(default=None, max_length=CURSOR_MAX_LENGTH),
    ):
        identity = actor(request, Scope.READ, require_signed_scope=True)
        return page(service.request_page, identity.subject_id, limit, cursor)

    @router.post("/friends/requests", status_code=201)
    def create_friend_request(body: FriendRequestBody, request: Request):
        identity = actor(request, Scope.PLAY)
        try:
            return service.request_friend(identity.subject_id, body.recipientId)
        except KeyError as exc:
            raise HTTPException(404, "user not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/friends/requests/{request_id}:respond")
    def respond_friend_request(request_id: str, body: FriendResponseBody, request: Request):
        identity = actor(request, Scope.PLAY)
        try:
            return service.respond_friend(request_id, identity.subject_id, accept=body.accept)
        except KeyError as exc:
            raise HTTPException(404, "friend request not found") from exc

    routes = APIRouter()
    routes.include_router(router)
    routes.include_router(paged_router)
    return routes
