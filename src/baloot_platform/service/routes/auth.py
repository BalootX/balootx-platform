from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from baloot_platform.identity import Identity, IdentityService
from baloot_platform.identity.pagination import (
    CURSOR_MAX_LENGTH,
    IdentityPaginationConfig,
    InvalidIdentityCursor,
)
from baloot_platform.service.client_identity import service_client_identity

COOKIE_NAME = "baloot_session"


class GuestRequest(BaseModel):
    displayName: str | None = Field(default=None, max_length=40)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)
    displayName: str | None = Field(default=None, max_length=40)


class LoginRequest(BaseModel):
    username: str
    password: str


class FriendRequestBody(BaseModel):
    recipientId: str


class FriendResponseBody(BaseModel):
    accept: bool = True


class IdentityPage(BaseModel):
    items: list[dict[str, Any]]
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
        role = str(getattr(claims, "role", "player"))
        synthetic = Identity(
            str(claims.subject_id), role,
            request.headers.get("x-baloot-subject-name", str(claims.subject_id))[:40],
            None, role in {"service", "admin", "referee"}, None,
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
        return service_client_identity(request.headers, "API Client"), {"csrf_token": ""}
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


def build_auth_router(service: IdentityService, *, pagination: IdentityPaginationConfig) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["identity"])

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
    def logout(request: Request, response: Response):
        service.logout(request_token(request))
        response.delete_cookie(COOKIE_NAME, path="/")
        return Response(status_code=204)

    @router.get("/users:search", response_model=IdentityPage)
    def search_users(
        request: Request, query: str, limit: int | None = Query(default=None, ge=1),
        cursor: str | None = Query(default=None, max_length=CURSOR_MAX_LENGTH),
    ):
        require_identity(request, service)
        if "\0" in query:
            raise HTTPException(422, [{"loc": ["query", "query"], "msg": "invalid search query", "type": "value_error"}])
        return page(service.search, query, limit, cursor)

    @router.get("/friends", response_model=IdentityPage)
    def friends(
        request: Request, limit: int | None = Query(default=None, ge=1),
        cursor: str | None = Query(default=None, max_length=CURSOR_MAX_LENGTH),
    ):
        identity, _ = require_identity(request, service)
        return page(service.friends, identity.subject_id, limit, cursor)

    @router.get("/friends/requests", response_model=IdentityPage)
    def friend_requests(
        request: Request, limit: int | None = Query(default=None, ge=1),
        cursor: str | None = Query(default=None, max_length=CURSOR_MAX_LENGTH),
    ):
        identity, _ = require_identity(request, service)
        return page(service.requests, identity.subject_id, limit, cursor)

    @router.post("/friends/requests", status_code=201)
    def create_friend_request(body: FriendRequestBody, request: Request):
        identity, _ = require_identity(request, service)
        try:
            return service.request_friend(identity.subject_id, body.recipientId)
        except KeyError as exc:
            raise HTTPException(404, "user not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/friends/requests/{request_id}:respond")
    def respond_friend_request(request_id: str, body: FriendResponseBody, request: Request):
        identity, _ = require_identity(request, service)
        try:
            return service.respond_friend(request_id, identity.subject_id, accept=body.accept)
        except KeyError as exc:
            raise HTTPException(404, "friend request not found") from exc

    return router
