from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterator, Literal
from urllib.parse import urlencode, urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from baloot_platform.version import PROFILE_VERSION_PATTERN

if TYPE_CHECKING:
    import jwt

    from baloot_platform.configuration import VersionedConfigurationService

OIDC_PROVIDER_FAMILY = "oidc_providers"
OIDC_PROFILE_ENV = "BALOOT_OIDC_PROVIDERS_PROFILE"
_LEGACY_PROVIDER_FIELDS = (
    "PROVIDER_ID",
    "ISSUER",
    "AUTHORIZATION_ENDPOINT",
    "TOKEN_ENDPOINT",
    "CLIENT_ID",
    "REDIRECT_URI",
    "JWKS_URI",
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
    allowed_algorithms: tuple[str, ...] = field(kw_only=True)
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
    allowed_algorithms: list[Literal["RS256", "ES256", "EdDSA"]] = Field(min_length=1)
    client_secret_env: str | None = Field(
        default=None, pattern=r"^BALOOT_OIDC_CLIENT_SECRET(?:_[A-Z0-9_]+)?$"
    )

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, value: str) -> str:
        if not value.strip() or not value.isprintable():
            raise ValueError("must be a non-blank, printable client ID")
        return value

    @field_validator("allowed_algorithms")
    @classmethod
    def validate_algorithms(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("must contain unique algorithms")
        return values

    @field_validator("issuer", "authorization_endpoint", "token_endpoint", "redirect_uri")
    @classmethod
    def validate_url(cls, value: str, info: ValidationInfo) -> str:
        _validate_oidc_url(value, allow_loopback=info.field_name == "redirect_uri")
        if info.field_name == "issuer" and "?" in value:
            raise ValueError("issuer must not contain a query")
        return value


def _validate_oidc_url(value: str, *, allow_loopback: bool = False) -> None:
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise ValueError("invalid URL authority") from None
    loopback_redirect = (
        allow_loopback and url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
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
        raise ValueError("must be an absolute HTTPS URL without user info or a fragment")


def discover_oidc_issuer(client: httpx.Client, issuer: str) -> tuple[str, str]:
    _validate_oidc_url(issuer)
    if "?" in issuer:
        raise ValueError("OIDC issuer must not contain a query")
    discovery_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        response = client.get(discovery_url, follow_redirects=False)
        response.raise_for_status()
        metadata = response.json()
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        raise ValueError("OIDC issuer discovery failed") from None
    if not isinstance(metadata, dict) or not isinstance(metadata.get("issuer"), str):
        raise ValueError("OIDC discovery requires an issuer")
    jwks_uri = metadata.get("jwks_uri")
    if not isinstance(jwks_uri, str):
        raise ValueError("OIDC discovery requires a jwks_uri")
    _validate_oidc_url(jwks_uri)
    return metadata["issuer"], jwks_uri


class _RegistryIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(pattern=PROFILE_VERSION_PATTERN)


class OidcProviderProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    profile: _RegistryIdentity
    change_reason: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    providers: dict[
        Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")],
        _OidcProviderRow,
    ]


def validate_oidc_provider_profile(payload: dict[str, Any]) -> OidcProviderProfile:
    try:
        return OidcProviderProfile.model_validate(payload)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            location = error["loc"]
            if len(location) >= 2 and location[0] == "providers":
                label = f"OIDC provider {location[1]!r}"
                if len(location) > 2 and error["type"] != "extra_forbidden":
                    label += f" {location[2]}"
            else:
                label = "OIDC registry"
                if location and error["type"] != "extra_forbidden":
                    label += " " + ".".join(map(str, location))
            problems.append(f"{label}: {error['msg']}")
        raise ValueError("; ".join(problems)) from None


def load_oidc_providers(configuration: VersionedConfigurationService) -> dict[str, OidcProviderConfig]:
    profile_id = os.environ.get(OIDC_PROFILE_ENV)
    has_legacy_metadata = any(
        f"BALOOT_OIDC_{name}" in os.environ for name in _LEGACY_PROVIDER_FIELDS
    )
    if profile_id is None:
        if has_legacy_metadata:
            raise ValueError(f"migrate legacy OIDC provider settings to {OIDC_PROFILE_ENV}")
        return {}
    try:
        revision = configuration.resolve_profile(
            family=OIDC_PROVIDER_FAMILY, profile_id=profile_id
        )
    except KeyError:
        raise ValueError(
            f"{OIDC_PROFILE_ENV} has no active published OIDC provider profile"
        ) from None
    if (
        revision["scope"] != "global"
        or revision["environment"] is not None
        or revision["effectiveFrom"] is not None
        or revision["effectiveUntil"] is not None
        or revision["rolloutPercentage"] != 100.0
    ):
        raise ValueError(
            f"{OIDC_PROFILE_ENV} requires a global provider profile without rollout conditions"
        )
    if has_legacy_metadata:
        logging.getLogger(__name__).warning(
            "Ignoring legacy OIDC provider metadata because a registry revision is configured"
        )

    profile = validate_oidc_provider_profile(revision["payload"])
    providers: dict[str, OidcProviderConfig] = {}
    for key, row in profile.providers.items():
        secret = None
        if row.client_secret_env is not None:
            secret = os.environ.get(row.client_secret_env)
            if not secret or not secret.strip():
                raise ValueError(
                    f"OIDC provider {key!r}: missing deployment secret {row.client_secret_env}"
                )
        providers[key] = OidcProviderConfig(
            provider_id=key,
            client_secret=secret,
            allowed_algorithms=tuple(row.allowed_algorithms),
            **row.model_dump(exclude={"client_secret_env", "allowed_algorithms"}),
        )
    logging.getLogger(__name__).info("Loaded OIDC provider profile revision %s", revision["revisionId"])
    return providers


@contextmanager
def http_oidc_runtime(
    configuration: VersionedConfigurationService, security_state: Any
) -> Iterator[OidcCallbackService | None]:
    providers = load_oidc_providers(configuration)
    if not providers:
        yield None
        return
    transactions = PkceTransactionStore(
        ttl_seconds=int(os.environ.get("BALOOT_OIDC_TRANSACTION_TTL", "600")),
        backend=security_state,
    )
    callback = OidcCallbackService(providers, transactions)
    try:
        yield callback
    finally:
        callback.close()


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
        self._jwks_clients: dict[str, jwt.PyJWKClient] = {}
        self._discovery_locks = {key: threading.Lock() for key in providers}

    def _jwks_client(self, provider: OidcProviderConfig) -> jwt.PyJWKClient:
        from jwt import PyJWKClient

        with self._discovery_locks[provider.provider_id]:
            cached = self._jwks_clients.get(provider.provider_id)
            if cached is not None:
                return cached
            issuer, jwks_uri = discover_oidc_issuer(self._client, provider.issuer)
            if issuer != provider.issuer:
                raise ValueError("OIDC discovery issuer mismatch")
            # PyJWT 2.15.1 rejects redirects when fetching the discovered key set.
            client = PyJWKClient(jwks_uri, timeout=15)
            self._jwks_clients[provider.provider_id] = client
            return client

    def _verify_id_token(self, token: str, provider: OidcProviderConfig, nonce: str) -> dict[str, Any]:
        if self.claims_verifier is not None:
            claims = dict(self.claims_verifier(token, provider, nonce))
        else:
            try:
                import jwt
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError('OIDC callback verification requires "baloot-platform[security]"') from exc
            try:
                header = jwt.get_unverified_header(token)
                if header.get("alg") not in provider.allowed_algorithms:
                    raise ValueError("OIDC ID token signing algorithm is not allowed")
                kid = header.get("kid")
                if not isinstance(kid, str) or not kid:
                    raise ValueError("OIDC ID token signing key ID is missing")
                signing_key = self._jwks_client(provider).get_signing_key(kid)
                claims = jwt.decode(
                    token,
                    signing_key,
                    algorithms=provider.allowed_algorithms,
                    audience=provider.client_id,
                    issuer=provider.issuer,
                    options={"require": ["exp", "iat", "iss", "aud", "sub"]},
                )
            except jwt.PyJWTError:
                raise ValueError("OIDC ID token validation failed") from None
        if str(claims.get("nonce") or "") != nonce:
            raise ValueError("OIDC nonce mismatch")
        if claims.get("iss") != provider.issuer:
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
