from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)


@dataclass(frozen=True, slots=True)
class OidcProviderConfig:
    provider_id: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    redirect_uri: str
    scopes: tuple[str, ...] = ("openid", "profile", "email")
    jwks_uri: str | None = None
    client_secret: str | None = field(default=None, repr=False)

    def authorization_url(self, *, state: str, nonce: str, code_challenge: str) -> str:
        separator = "&" if "?" in self.authorization_endpoint else "?"
        return f"{self.authorization_endpoint}{separator}{urlencode({
            'response_type': 'code', 'client_id': self.client_id, 'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes), 'state': state, 'nonce': nonce,
            'code_challenge': code_challenge, 'code_challenge_method': 'S256',
        })}"


class _OidcProviderRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    redirect_uri: str
    jwks_uri: str | None = None
    client_secret_env: str | None = Field(
        default=None, pattern=r"^BALOOT_OIDC_CLIENT_SECRET(?:_[A-Z0-9_]+)?$"
    )

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, value: str) -> str:
        if not value.strip() or not value.isprintable():
            raise ValueError("must be a non-blank, printable client ID")
        return value

    @field_validator(
        "issuer", "authorization_endpoint", "token_endpoint", "redirect_uri", "jwks_uri"
    )
    @classmethod
    def validate_url(cls, value: str, info: ValidationInfo) -> str:
        try:
            url = urlsplit(value)
            port = url.port
        except ValueError:
            raise ValueError("invalid URL authority") from None
        loopback_redirect = (
            info.field_name == "redirect_uri"
            and url.scheme == "http"
            and url.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        if (
            not (url.scheme == "https" or loopback_redirect)
            or not url.hostname
            or port == 0
            or "@" in url.netloc
            or "#" in value
            or "\\" in value
            or not value.isascii()
            or any(not char.isprintable() or char.isspace() for char in value)
        ):
            raise ValueError("must be an absolute HTTPS URL (HTTP loopback redirect allowed)")
        if info.field_name == "issuer" and "?" in value:
            raise ValueError("issuer must not contain a query")
        return value


def load_oidc_providers() -> dict[str, OidcProviderConfig]:
    path = os.environ.get("BALOOT_OIDC_PROVIDERS_FILE")
    if path is None:
        legacy_fields = (
            "PROVIDER_ID",
            "ISSUER",
            "AUTHORIZATION_ENDPOINT",
            "TOKEN_ENDPOINT",
            "CLIENT_ID",
            "REDIRECT_URI",
            "JWKS_URI",
        )
        if any(f"BALOOT_OIDC_{name}" in os.environ for name in legacy_fields):
            raise ValueError("migrate legacy OIDC provider settings to BALOOT_OIDC_PROVIDERS_FILE")
        return {}

    try:
        with Path(path).open("rb") as config:
            rows = tomllib.load(config)
    except OSError:
        raise ValueError("BALOOT_OIDC_PROVIDERS_FILE must name a readable file") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ValueError("BALOOT_OIDC_PROVIDERS_FILE must contain valid UTF-8 TOML") from None

    providers: dict[str, OidcProviderConfig] = {}
    for key, fields in rows.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ValueError(f"OIDC provider {key!r}: invalid provider key")
        try:
            row = _OidcProviderRow.model_validate(fields)
        except ValidationError as exc:
            problems = []
            for error in exc.errors(
                include_input=False, include_context=False, include_url=False
            ):
                if error["type"] == "extra_forbidden":
                    problems.append("unsupported field")
                elif error["loc"]:
                    problems.append(f"{error['loc'][0]}: {error['msg']}")
                else:
                    problems.append("expected a provider table")
            raise ValueError(f"OIDC provider {key!r}: {', '.join(problems)}") from None
        secret = None
        if row.client_secret_env is not None:
            secret = os.environ.get(row.client_secret_env)
            if not secret or not secret.strip():
                raise ValueError(
                    f"OIDC provider {key!r}: missing deployment secret {row.client_secret_env}"
                )
        providers[key] = OidcProviderConfig(
            provider_id=key, client_secret=secret, **row.model_dump(exclude={"client_secret_env"})
        )
    return providers


@dataclass(frozen=True, slots=True)
class PkceTransaction:
    transaction_id: str
    provider_id: str
    state: str
    nonce: str
    verifier: str
    challenge: str
    created_at: int
    expires_at: int
    return_to: str


class PkceTransactionStore:
    def __init__(self, *, ttl_seconds: int = 600, backend: Any | None = None):
        self.ttl_seconds = max(60, min(1800, int(ttl_seconds)))
        self.backend = backend
        self._items: dict[str, PkceTransaction] = {}

    @staticmethod
    def _challenge(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def create(self, provider_id: str, *, return_to: str = "/") -> PkceTransaction:
        now = int(time.time())
        verifier = secrets.token_urlsafe(48)
        item = PkceTransaction(
            transaction_id=secrets.token_urlsafe(18), provider_id=provider_id,
            state=secrets.token_urlsafe(24), nonce=secrets.token_urlsafe(24),
            verifier=verifier, challenge=self._challenge(verifier), created_at=now,
            expires_at=now + self.ttl_seconds, return_to=return_to,
        )
        if self.backend is not None:
            self.backend.save_oidc(item.state, asdict(item), item.expires_at)
        else:
            self._items[item.state] = item
        return item

    def consume(self, state: str, *, now: int | None = None) -> PkceTransaction:
        current = int(time.time() if now is None else now)
        if self.backend is not None:
            payload = self.backend.consume_oidc(state, current)
            item = PkceTransaction(**payload) if payload else None
        else:
            item = self._items.pop(state, None)
        if item is None:
            raise ValueError("unknown or reused OIDC state")
        if item.expires_at <= current:
            raise ValueError("OIDC transaction expired")
        return item


class OidcCallbackService:
    """Completes Authorization Code + PKCE and validates ID-token claims.

    Signature verification uses PyJWT/JWKS when installed. A verifier callback
    can be injected for hermetic tests and enterprise identity adapters.
    """

    def __init__(self, providers: dict[str, OidcProviderConfig], transactions: PkceTransactionStore,
                 *, http_client: httpx.Client | None = None,
                 claims_verifier: Callable[[str, OidcProviderConfig, str], dict[str, Any]] | None = None):
        self.providers = dict(providers)
        self.transactions = transactions
        self._client = http_client or httpx.Client(timeout=15.0, follow_redirects=False)
        self._owns_client = http_client is None
        self.claims_verifier = claims_verifier

    def _verify_id_token(self, token: str, provider: OidcProviderConfig, nonce: str) -> dict[str, Any]:
        if self.claims_verifier is not None:
            claims = dict(self.claims_verifier(token, provider, nonce))
        else:
            try:
                import jwt
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError('OIDC callback verification requires "baloot-platform[security]"') from exc
            jwks_uri = provider.jwks_uri or provider.issuer.rstrip("/") + "/.well-known/jwks.json"
            signing_key = jwt.PyJWKClient(jwks_uri).get_signing_key_from_jwt(token).key
            claims = dict(jwt.decode(
                token, signing_key, algorithms=["RS256", "ES256", "EdDSA"],
                audience=provider.client_id, issuer=provider.issuer,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            ))
        if str(claims.get("nonce") or "") != nonce:
            raise ValueError("OIDC nonce mismatch")
        if str(claims.get("iss") or "").rstrip("/") != provider.issuer.rstrip("/"):
            raise ValueError("OIDC issuer mismatch")
        aud = claims.get("aud")
        audiences = {str(item) for item in aud} if isinstance(aud, list) else {str(aud)}
        if provider.client_id not in audiences:
            raise ValueError("OIDC audience mismatch")
        if not claims.get("sub"):
            raise ValueError("OIDC subject is missing")
        return claims

    def complete(self, *, provider_id: str, state: str, code: str) -> dict[str, Any]:
        provider = self.providers.get(provider_id)
        if provider is None:
            raise ValueError("OIDC provider is not configured")
        transaction = self.transactions.consume(state)
        if transaction.provider_id != provider_id:
            raise ValueError("OIDC provider/state mismatch")
        body: dict[str, Any] = {
            "grant_type": "authorization_code", "code": code,
            "redirect_uri": provider.redirect_uri, "client_id": provider.client_id,
            "code_verifier": transaction.verifier,
        }
        if provider.client_secret:
            body["client_secret"] = provider.client_secret
        response = self._client.post(provider.token_endpoint, data=body, headers={"Accept": "application/json"})
        response.raise_for_status()
        tokens = dict(response.json())
        id_token = str(tokens.get("id_token") or "")
        if not id_token:
            raise ValueError("OIDC provider did not return an ID token")
        claims = self._verify_id_token(id_token, provider, transaction.nonce)
        return {
            "providerId": provider_id, "issuer": provider.issuer,
            "externalSubject": str(claims["sub"]), "claims": claims,
            "returnTo": transaction.return_to,
            "accessTokenPresent": bool(tokens.get("access_token")),
        }

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
