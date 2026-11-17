from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping


class Scope(StrEnum):
    READ = "baloot.read"
    PLAY = "baloot.play"
    ANALYZE = "baloot.analyze"
    LIVE = "baloot.live"
    HIDDEN_READ = "baloot.hidden.read"
    REFEREE = "baloot.referee"
    CONFIG_ADMIN = "baloot.config.admin"
    MODEL_ADMIN = "baloot.model.admin"


ROLE_SCOPES: dict[str, frozenset[Scope]] = {
    "guest": frozenset({Scope.READ, Scope.PLAY, Scope.LIVE}),
    "player": frozenset({Scope.READ, Scope.PLAY, Scope.ANALYZE, Scope.LIVE}),
    "expert": frozenset({Scope.READ, Scope.PLAY, Scope.ANALYZE, Scope.LIVE}),
    "referee": frozenset({Scope.READ, Scope.PLAY, Scope.ANALYZE, Scope.LIVE, Scope.REFEREE, Scope.HIDDEN_READ}),
    "admin": frozenset(Scope),
    "service": frozenset(Scope),
}


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True, slots=True)
class AccessClaims:
    subject_id: str
    audience: str
    issuer: str
    scopes: tuple[str, ...]
    issued_at: int
    expires_at: int
    token_id: str
    tenant_id: str | None = None
    role: str = "player"
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.subject_id,
            "aud": self.audience,
            "iss": self.issuer,
            "scope": " ".join(self.scopes),
            "iat": self.issued_at,
            "exp": self.expires_at,
            "jti": self.token_id,
            "tenant": self.tenant_id,
            "role": self.role,
            "sid": self.session_id,
        }


class SignedAccessTokenService:
    """Small deterministic HMAC token boundary for local/service deployments.

    Production OIDC access tokens may replace this implementation. The rest of
    the platform consumes AccessClaims and AuthorizationPolicy only.
    """

    def __init__(self, secret: bytes, *, issuer: str = "baloot-platform", audience: str = "baloot-api"):
        if len(secret) < 32:
            raise ValueError("token secret must be at least 32 bytes")
        self.secret = bytes(secret)
        self.issuer = issuer
        self.audience = audience

    def issue(self, *, subject_id: str, scopes: Iterable[Scope | str], lifetime_seconds: int = 900,
              tenant_id: str | None = None, role: str = "player", session_id: str | None = None,
              now: int | None = None) -> str:
        issued = int(time.time() if now is None else now)
        normalized = tuple(sorted({str(Scope(item).value) for item in scopes}))
        claims = AccessClaims(
            subject_id=subject_id, audience=self.audience, issuer=self.issuer, scopes=normalized,
            issued_at=issued, expires_at=issued + max(30, min(86400, int(lifetime_seconds))),
            token_id=secrets.token_urlsafe(18), tenant_id=tenant_id, role=role, session_id=session_id,
        )
        header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":"), sort_keys=True).encode())
        body = _b64(json.dumps(claims.to_dict(), separators=(",", ":"), sort_keys=True).encode())
        signature = _b64(hmac.new(self.secret, f"{header}.{body}".encode(), hashlib.sha256).digest())
        return f"{header}.{body}.{signature}"

    def verify(self, token: str, *, required_audience: str | None = None, now: int | None = None) -> AccessClaims:
        try:
            header, body, signature = token.split(".", 2)
        except ValueError as exc:
            raise ValueError("invalid access token") from exc
        expected = _b64(hmac.new(self.secret, f"{header}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid access token signature")
        data = json.loads(_unb64(body))
        current = int(time.time() if now is None else now)
        if str(data.get("iss")) != self.issuer:
            raise ValueError("invalid token issuer")
        audience = str(data.get("aud"))
        if audience != str(required_audience or self.audience):
            raise ValueError("invalid token audience")
        if int(data.get("exp", 0)) <= current:
            raise ValueError("access token expired")
        if int(data.get("iat", 0)) > current + 60:
            raise ValueError("access token issued in the future")
        scopes = tuple(filter(None, str(data.get("scope", "")).split()))
        return AccessClaims(
            subject_id=str(data["sub"]), audience=audience, issuer=str(data["iss"]), scopes=scopes,
            issued_at=int(data["iat"]), expires_at=int(data["exp"]), token_id=str(data["jti"]),
            tenant_id=str(data["tenant"]) if data.get("tenant") else None,
            role=str(data.get("role") or "player"), session_id=str(data["sid"]) if data.get("sid") else None,
        )


class AuthorizationPolicy:
    def __init__(self, role_scopes: Mapping[str, Iterable[Scope | str]] | None = None):
        source = role_scopes or ROLE_SCOPES
        self.role_scopes = {
            str(role): frozenset(str(Scope(item).value) for item in scopes)
            for role, scopes in source.items()
        }

    def effective_scopes(self, *, role: str, claims: AccessClaims | None = None) -> frozenset[str]:
        role_values = self.role_scopes.get(role, frozenset())
        if claims is None:
            return role_values
        claimed = frozenset(claims.scopes)
        return role_values & claimed if role_values else claimed

    def require(self, *, role: str, required: Scope | str, claims: AccessClaims | None = None) -> None:
        value = str(Scope(required).value)
        if value not in self.effective_scopes(role=role, claims=claims):
            raise PermissionError(f"missing required scope: {value}")

    def can_access_resource(self, *, subject_id: str, owner_subject_id: str | None,
                            role: str, required: Scope | str, claims: AccessClaims | None = None,
                            allow_public: bool = False, public: bool = False) -> bool:
        try:
            self.require(role=role, required=required, claims=claims)
        except PermissionError:
            return False
        if public and allow_public:
            return True
        if role in {"admin", "referee", "service"}:
            return True
        return owner_subject_id is not None and subject_id == owner_subject_id
