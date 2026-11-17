from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import urlencode

import httpx


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
    client_secret: str | None = None

    def authorization_url(self, *, state: str, nonce: str, code_challenge: str) -> str:
        return f"{self.authorization_endpoint}?{urlencode({
            'response_type': 'code', 'client_id': self.client_id, 'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes), 'state': state, 'nonce': nonce,
            'code_challenge': code_challenge, 'code_challenge_method': 'S256',
        })}"


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
