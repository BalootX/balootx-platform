from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from baloot_platform.identity import IdentityService
from baloot_platform.security import (
    AuthorizationPolicy, PasskeyChallengeService, ROLE_SCOPES, Scope,
    SignedAccessTokenService, WebSocketTicketService,
)

from .auth import require_identity


def _current_oidc(request: Request):
    runtime = getattr(request.app.state, "oidc", None)
    if runtime is None:
        return None
    try:
        return runtime.current()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(503, "OIDC provider configuration is unavailable") from exc


class TokenRequest(BaseModel):
    scopes: list[str] = Field(default_factory=lambda: [Scope.READ.value, Scope.PLAY.value])
    lifetimeSeconds: int = Field(900, ge=30, le=86400)
    role: str = "player"


class OidcBeginRequest(BaseModel):
    providerId: str
    returnTo: str = "/"


class OidcCallbackRequest(BaseModel):
    providerId: str
    state: str
    code: str


class PasskeyBeginRequest(BaseModel):
    purpose: str


class PasskeyRegistrationFinishRequest(BaseModel):
    challenge: str
    payload: dict[str, Any]
    label: str = "Passkey"


class PasskeyAuthenticationFinishRequest(BaseModel):
    challenge: str
    credentialId: str
    payload: dict[str, Any]


class WebSocketTicketRequest(BaseModel):
    roomId: str
    tenantId: str | None = None


def build_security_router(
    identities: IdentityService,
    tokens: SignedAccessTokenService,
    policy: AuthorizationPolicy,
    passkeys: PasskeyChallengeService,
    websocket_tickets: WebSocketTicketService,
    security_state: Any,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/security", tags=["security"])

    @router.get("/capabilities")
    def capabilities(request: Request):
        oidc = _current_oidc(request)
        return {
            "optionalSignIn": True, "localGuest": True, "localAccount": True,
            "oidcProviders": sorted(oidc.providers) if oidc else [], "oidcCallback": True,
            "passkeys": True, "websocketTickets": True,
            "roles": {key: sorted(value.value for value in values) for key, values in ROLE_SCOPES.items()},
        }

    @router.post("/tokens")
    def issue_token(body: TokenRequest, request: Request):
        identity, session = require_identity(request, identities)
        role = body.role.strip().lower()
        if role not in ROLE_SCOPES:
            raise HTTPException(422, "unknown role")
        if role in {"admin", "referee", "service"} and not bool(getattr(request.state, "api_key_valid", False)):
            raise HTTPException(403, "privileged roles require service authorization")
        try:
            requested = [Scope(item) for item in body.scopes]
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        allowed = policy.role_scopes[role]
        if any(item.value not in allowed for item in requested):
            raise HTTPException(403, "requested scope is not allowed for role")
        access_token = tokens.issue(
            subject_id=identity.subject_id, scopes=requested, lifetime_seconds=body.lifetimeSeconds,
            role=role, session_id=str(session.get("session_id") or session.get("sessionId") or ""),
        )
        return {"accessToken": access_token, "tokenType": "Bearer", "expiresIn": body.lifetimeSeconds, "role": role}

    @router.post("/oidc:begin")
    def oidc_begin(body: OidcBeginRequest, request: Request):
        oidc = _current_oidc(request)
        provider = oidc.providers.get(body.providerId) if oidc else None
        if provider is None:
            raise HTTPException(404, "OIDC provider is not configured")
        transaction = oidc.transactions.create(
            provider.provider_id, return_to=body.returnTo,
            provider_binding=provider.transaction_binding(),
        )
        return {
            "authorizationUrl": provider.authorization_url(
                state=transaction.state, nonce=transaction.nonce, code_challenge=transaction.challenge,
            ),
            "transactionId": transaction.transaction_id, "state": transaction.state,
            "expiresAt": transaction.expires_at,
        }

    @router.post("/oidc:callback")
    def oidc_complete(body: OidcCallbackRequest, request: Request):
        oidc = _current_oidc(request)
        if oidc is None:
            raise HTTPException(400, "OIDC provider is not configured")
        try:
            result = oidc.complete(provider_id=body.providerId, state=body.state, code=body.code)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        existing = security_state.get_external_identity(result["providerId"], result["issuer"], result["externalSubject"])
        claims = dict(result["claims"])
        display = str(claims.get("name") or claims.get("preferred_username") or claims.get("email") or "مستخدم البلوت")[:40]
        local_subject = str(existing["local_subject_id"]) if existing else None
        session = identities.create_external(kind="oidc", display_name=display, subject_id=local_subject,
                                             user_agent=request.headers.get("user-agent"))
        security_state.link_external_identity(
            provider_id=result["providerId"], issuer=result["issuer"], external_subject=result["externalSubject"],
            local_subject_id=session["identity"]["subjectId"], claims=claims,
            linked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        return {**session, "returnTo": result["returnTo"], "providerId": result["providerId"]}

    @router.post("/passkeys:begin")
    def passkey_begin(body: PasskeyBeginRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return passkeys.begin(subject_id=identity.subject_id, purpose=body.purpose)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/passkeys:register:finish")
    def passkey_register_finish(body: PasskeyRegistrationFinishRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            item = passkeys.finish_registration(challenge=body.challenge, subject_id=identity.subject_id,
                                                payload=body.payload, label=body.label)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"credentialId": item.credential_id, "createdAt": item.created_at, "label": item.label}

    @router.post("/passkeys:authenticate:finish")
    def passkey_auth_finish(body: PasskeyAuthenticationFinishRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            item = passkeys.finish_authentication(challenge=body.challenge, subject_id=identity.subject_id,
                                                  credential_id=body.credentialId, payload=body.payload)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"credentialId": item.credential_id, "lastUsedAt": item.last_used_at, "signCount": item.sign_count}

    @router.get("/passkeys")
    def list_passkeys(request: Request):
        identity, _ = require_identity(request, identities)
        return {"items": passkeys.list_credentials(identity.subject_id)}

    @router.post("/websocket-tickets")
    def websocket_ticket(body: WebSocketTicketRequest, request: Request):
        identity, _ = require_identity(request, identities)
        return websocket_tickets.issue(subject_id=identity.subject_id, room_id=body.roomId,
                                       tenant_id=body.tenantId, scopes=(Scope.LIVE.value,))

    return router
