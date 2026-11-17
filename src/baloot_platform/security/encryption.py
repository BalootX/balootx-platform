from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class PayloadCipher(Protocol):
    key_id: str
    def encrypt_json(self, value: Mapping[str, Any], *, aad: bytes = b"") -> str: ...
    def decrypt_json(self, value: str, *, aad: bytes = b"") -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AesGcmPayloadCipher:
    """AES-256-GCM envelope for trusted hidden-state payloads."""

    key: bytes
    key_id: str = "local-v1"

    def __post_init__(self) -> None:
        if len(self.key) != 32:
            raise ValueError("AES-GCM key must be exactly 32 bytes")

    @classmethod
    def from_secret(cls, secret: bytes | str, *, key_id: str = "local-v1") -> "AesGcmPayloadCipher":
        raw = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        return cls(hashlib.sha256(raw).digest(), key_id)

    @classmethod
    def from_environment(cls, name: str = "BALOOT_HIDDEN_STATE_KEY") -> "AesGcmPayloadCipher | None":
        value = os.environ.get(name, "")
        return None if not value else cls.from_secret(value, key_id=os.environ.get("BALOOT_HIDDEN_STATE_KEY_ID", "env-v1"))

    def encrypt_json(self, value: Mapping[str, Any], *, aad: bytes = b"") -> str:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        body = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ciphertext = AESGCM(self.key).encrypt(nonce, body, aad)
        envelope = {
            "schema": "baloot.encrypted-json",
            "version": 1,
            "algorithm": "AES-256-GCM",
            "keyId": self.key_id,
            "nonce": base64.urlsafe_b64encode(nonce).decode("ascii").rstrip("="),
            "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii").rstrip("="),
        }
        return "enc:v1:" + base64.urlsafe_b64encode(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii").rstrip("=")

    def decrypt_json(self, value: str, *, aad: bytes = b"") -> dict[str, Any]:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if not value.startswith("enc:v1:"):
            raise ValueError("unsupported encrypted payload")
        token = value[7:]
        envelope = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
        if envelope.get("keyId") != self.key_id:
            raise ValueError("encrypted payload key identity mismatch")
        nonce_text = str(envelope["nonce"])
        cipher_text = str(envelope["ciphertext"])
        nonce = base64.urlsafe_b64decode(nonce_text + "=" * (-len(nonce_text) % 4))
        ciphertext = base64.urlsafe_b64decode(cipher_text + "=" * (-len(cipher_text) % 4))
        plaintext = AESGCM(self.key).decrypt(nonce, ciphertext, aad)
        result = json.loads(plaintext)
        if not isinstance(result, dict):
            raise ValueError("decrypted payload must be an object")
        return result
