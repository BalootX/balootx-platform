from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from baloot_platform.identity.ownership import owns_subject


@dataclass(frozen=True, slots=True)
class PasskeyCredential:
    credential_id: str
    subject_id: str
    public_key_cose: bytes
    sign_count: int
    transports: tuple[str, ...]
    created_at: int
    last_used_at: int | None = None
    label: str = "Passkey"


class WebAuthnAdapter(Protocol):
    def registration_options(self, *, subject_id: str, challenge: str, rp_id: str) -> dict[str, Any]: ...
    def verify_registration(self, payload: dict[str, Any], *, expected_challenge: str,
                            expected_origin: str, expected_rp_id: str) -> dict[str, Any]: ...
    def authentication_options(self, *, subject_id: str, challenge: str, rp_id: str,
                               credentials: list[dict[str, Any]]) -> dict[str, Any]: ...
    def verify_authentication(self, payload: dict[str, Any], *, expected_challenge: str,
                              expected_origin: str, expected_rp_id: str,
                              credential: PasskeyCredential) -> dict[str, Any]: ...


class PasskeyChallengeService:
    """Durable WebAuthn ceremony state with a provider-neutral verifier."""

    def __init__(self, *, rp_id: str, origin: str, ttl_seconds: int = 300,
                 backend: Any | None = None, adapter: WebAuthnAdapter | None = None):
        self.rp_id = rp_id
        self.origin = origin.rstrip("/")
        self.ttl_seconds = max(60, min(900, int(ttl_seconds)))
        self.backend = backend
        self.adapter = adapter
        self._challenges: dict[str, dict[str, Any]] = {}
        self._credentials: dict[str, PasskeyCredential] = {}

    def begin(self, *, subject_id: str, purpose: str) -> dict[str, Any]:
        if purpose not in {"register", "authenticate"}:
            raise ValueError("unsupported passkey purpose")
        challenge = secrets.token_urlsafe(32)
        now = int(time.time())
        digest = hashlib.sha256(challenge.encode()).hexdigest()
        payload = {"subject_id": subject_id, "purpose": purpose, "challenge": challenge,
                   "created_at": now, "expires_at": now + self.ttl_seconds}
        if self.backend is not None:
            self.backend.save_challenge(digest, subject_id, purpose, payload, payload["expires_at"])
        else:
            self._challenges[digest] = payload
        base = {"challenge": challenge, "rpId": self.rp_id, "origin": self.origin,
                "purpose": purpose, "expiresAt": payload["expires_at"]}
        if self.adapter is None:
            return base
        if purpose == "register":
            return {**base, "publicKey": self.adapter.registration_options(subject_id=subject_id, challenge=challenge, rp_id=self.rp_id)}
        return {**base, "publicKey": self.adapter.authentication_options(
            subject_id=subject_id, challenge=challenge, rp_id=self.rp_id,
            credentials=self.list_credentials(subject_id),
        )}

    def consume(self, challenge: str, *, subject_id: str, purpose: str, now: int | None = None) -> dict[str, Any]:
        digest = hashlib.sha256(challenge.encode()).hexdigest()
        current = int(time.time() if now is None else now)
        if self.backend is not None:
            item = self.backend.consume_challenge(digest, subject_id, purpose, current)
        else:
            item = self._challenges.pop(digest, None)
        if item is None:
            raise ValueError("unknown, expired or reused passkey challenge")
        if int(item["expires_at"]) <= current or not owns_subject(subject_id, item["subject_id"]) or item["purpose"] != purpose:
            raise ValueError("passkey challenge binding mismatch")
        return item

    def _put(self, item: PasskeyCredential) -> None:
        if self.backend is not None:
            self.backend.put_credential(asdict(item))
        else:
            self._credentials[item.credential_id] = item

    def _get(self, credential_id: str) -> PasskeyCredential | None:
        if self.backend is not None:
            value = self.backend.get_credential(credential_id)
            return PasskeyCredential(**value) if value else None
        return self._credentials.get(credential_id)

    def register_verified(self, *, challenge: str, subject_id: str, credential_id: str,
                          public_key_cose: bytes, sign_count: int = 0,
                          transports: tuple[str, ...] = (), label: str = "Passkey") -> PasskeyCredential:
        self.consume(challenge, subject_id=subject_id, purpose="register")
        if self._get(credential_id) is not None:
            raise ValueError("credential already registered")
        item = PasskeyCredential(credential_id, subject_id, bytes(public_key_cose), max(0, int(sign_count)),
                                 tuple(transports), int(time.time()), None, label[:80])
        self._put(item)
        return item

    def finish_registration(self, *, challenge: str, subject_id: str, payload: dict[str, Any], label: str = "Passkey") -> PasskeyCredential:
        if self.adapter is None:
            raise RuntimeError("no WebAuthn adapter is configured")
        self.consume(challenge, subject_id=subject_id, purpose="register")
        verified = self.adapter.verify_registration(payload, expected_challenge=challenge,
                                                    expected_origin=self.origin, expected_rp_id=self.rp_id)
        item = PasskeyCredential(
            str(verified["credential_id"]), subject_id, bytes(verified["public_key_cose"]),
            max(0, int(verified.get("sign_count", 0))), tuple(verified.get("transports") or ()),
            int(time.time()), None, label[:80],
        )
        if self._get(item.credential_id) is not None:
            raise ValueError("credential already registered")
        self._put(item)
        return item

    def record_verified_assertion(self, *, challenge: str, subject_id: str,
                                  credential_id: str, sign_count: int) -> PasskeyCredential:
        self.consume(challenge, subject_id=subject_id, purpose="authenticate")
        item = self._get(credential_id)
        if item is None or not owns_subject(subject_id, item.subject_id):
            raise ValueError("unknown passkey credential")
        if item.sign_count and int(sign_count) <= item.sign_count:
            raise ValueError("passkey sign counter did not advance")
        updated = PasskeyCredential(item.credential_id, item.subject_id, item.public_key_cose,
                                    max(item.sign_count, int(sign_count)), item.transports,
                                    item.created_at, int(time.time()), item.label)
        self._put(updated)
        return updated

    def finish_authentication(self, *, challenge: str, subject_id: str,
                              credential_id: str, payload: dict[str, Any]) -> PasskeyCredential:
        if self.adapter is None:
            raise RuntimeError("no WebAuthn adapter is configured")
        item = self._get(credential_id)
        if item is None or not owns_subject(subject_id, item.subject_id):
            raise ValueError("unknown passkey credential")
        self.consume(challenge, subject_id=subject_id, purpose="authenticate")
        verified = self.adapter.verify_authentication(
            payload, expected_challenge=challenge, expected_origin=self.origin,
            expected_rp_id=self.rp_id, credential=item,
        )
        updated = PasskeyCredential(item.credential_id, item.subject_id, item.public_key_cose,
                                    max(item.sign_count, int(verified.get("sign_count", 0))),
                                    item.transports, item.created_at, int(time.time()), item.label)
        self._put(updated)
        return updated

    def list_credentials(self, subject_id: str) -> list[dict[str, Any]]:
        if self.backend is not None:
            values = self.backend.list_credentials(subject_id)
            items = [PasskeyCredential(**value) for value in values]
        else:
            items = [item for item in self._credentials.values() if owns_subject(subject_id, item.subject_id)]
        return [{"credentialId": item.credential_id, "subjectId": item.subject_id,
                 "signCount": item.sign_count, "transports": list(item.transports),
                 "createdAt": item.created_at, "lastUsedAt": item.last_used_at,
                 "label": item.label} for item in items]
