from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True, slots=True)
class WebSocketTicketClaims:
    subject_id: str
    room_id: str
    audience: str
    tenant_id: str | None
    scopes: tuple[str, ...]
    issued_at: int
    expires_at: int
    nonce: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjectId": self.subject_id,
            "roomId": self.room_id,
            "audience": self.audience,
            "tenantId": self.tenant_id,
            "scopes": list(self.scopes),
            "issuedAt": self.issued_at,
            "expiresAt": self.expires_at,
            "nonce": self.nonce,
        }


class WebSocketTicketService:
    """Issues one-time, short-lived, room-bound WebSocket tickets.

    URL query strings may contain the opaque ticket because it is single-use,
    audience-bound and short-lived. Long-lived bearer/session tokens are never
    accepted through the WebSocket URL.
    """

    def __init__(self, secret: bytes, store: Any, *, audience: str = "baloot-live-websocket",
                 lifetime_seconds: int = 45):
        if len(secret) < 32:
            raise ValueError("websocket ticket secret must contain at least 32 bytes")
        self._secret = bytes(secret)
        self.store = store
        self.audience = audience
        self.lifetime_seconds = max(10, min(120, int(lifetime_seconds)))

    def _sign(self, body: str) -> str:
        return _b64(hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest())

    def issue(self, *, subject_id: str, room_id: str, tenant_id: str | None = None,
              scopes: tuple[str, ...] = ("baloot.live",), now: int | None = None) -> dict[str, Any]:
        issued = int(time.time() if now is None else now)
        nonce = secrets.token_urlsafe(24)
        payload = {
            "sub": subject_id,
            "room": room_id,
            "aud": self.audience,
            "tenant": tenant_id,
            "scopes": sorted(set(str(item) for item in scopes)),
            "iat": issued,
            "exp": issued + self.lifetime_seconds,
            "nonce": nonce,
        }
        encoded = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        token = f"wst1.{encoded}.{self._sign(encoded)}"
        self.store.save_ticket(
            hashlib.sha256(nonce.encode("utf-8")).hexdigest(), subject_id, room_id,
            self.audience, payload["exp"],
        )
        return {"ticket": token, "tokenType": "WebSocketTicket", "expiresAt": payload["exp"], "audience": self.audience}

    def consume(self, token: str, *, room_id: str, audience: str | None = None,
                now: int | None = None) -> WebSocketTicketClaims:
        try:
            prefix, encoded, signature = token.split(".", 2)
        except ValueError as exc:
            raise ValueError("malformed WebSocket ticket") from exc
        if prefix != "wst1" or not hmac.compare_digest(signature, self._sign(encoded)):
            raise ValueError("invalid WebSocket ticket signature")
        try:
            payload = json.loads(_unb64(encoded))
        except Exception as exc:
            raise ValueError("invalid WebSocket ticket payload") from exc
        current = int(time.time() if now is None else now)
        expected_audience = audience or self.audience
        if payload.get("aud") != expected_audience:
            raise ValueError("WebSocket ticket audience mismatch")
        if payload.get("room") != room_id:
            raise ValueError("WebSocket ticket room mismatch")
        if int(payload.get("exp", 0)) <= current or int(payload.get("iat", 0)) > current + 30:
            raise ValueError("WebSocket ticket expired or not yet valid")
        nonce = str(payload.get("nonce") or "")
        consumed = self.store.consume_ticket(hashlib.sha256(nonce.encode("utf-8")).hexdigest(), current)
        if consumed is None:
            raise ValueError("WebSocket ticket is unknown, expired or already used")
        if consumed["subject_id"] != payload.get("sub") or consumed["room_id"] != room_id or consumed["audience"] != expected_audience:
            raise ValueError("WebSocket ticket storage binding mismatch")
        return WebSocketTicketClaims(
            subject_id=str(payload["sub"]), room_id=room_id, audience=expected_audience,
            tenant_id=str(payload["tenant"]) if payload.get("tenant") else None,
            scopes=tuple(str(item) for item in payload.get("scopes") or ()),
            issued_at=int(payload["iat"]), expires_at=int(payload["exp"]), nonce=nonce,
        )
