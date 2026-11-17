from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import SakkaBoard, canonical


class BoardSigner(Protocol):
    algorithm: str
    key_id: str

    def sign(self, board: SakkaBoard) -> str: ...
    def verify(self, board: SakkaBoard) -> bool: ...


def board_signature_payload(board: SakkaBoard) -> bytes:
    """Canonical, renderer- and database-independent board authentication payload."""
    return canonical({
        "schema": 1,
        "boardId": board.board_id,
        "version": board.version,
        "boardHash": board.board_hash,
        "serial": board.serial,
        "bundleHash": board.bundle_hash,
        "trustedPayloadHash": board.trusted_payload_hash,
        "publicEventPrefixHash": board.public_event_prefix_hash,
        "comparisonKey": board.comparison_key,
        "createdBy": board.created_by,
        "createdAt": board.created_at,
    })


@dataclass(frozen=True, slots=True)
class HmacBoardSigner:
    secret: bytes
    key_id: str
    algorithm: str = "hmac-sha256"

    @classmethod
    def from_secret(cls, secret: bytes) -> "HmacBoardSigner":
        if len(secret) < 32:
            raise ValueError("Sakka signing secret must contain at least 32 bytes")
        return cls(secret=bytes(secret), key_id=hashlib.sha256(secret).hexdigest()[:20])

    @classmethod
    def load_or_create(cls, runtime: str | Path, *, environment_name: str = "BALOOT_SAKKA_SIGNING_KEY") -> "HmacBoardSigner":
        configured = os.environ.get(environment_name, "").strip()
        if configured:
            # Accept a plain high-entropy secret or a 64-character hexadecimal key.
            try:
                secret = bytes.fromhex(configured) if len(configured) == 64 else configured.encode("utf-8")
            except ValueError as exc:
                raise ValueError(f"{environment_name} is not a valid signing key") from exc
            return cls.from_secret(secret)
        directory = Path(runtime)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "sakka-board-signing.key"
        if not path.exists():
            secret = secrets.token_bytes(32)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(path, flags, 0o600)
            try:
                os.write(descriptor, secret.hex().encode("ascii"))
            finally:
                os.close(descriptor)
        else:
            secret = bytes.fromhex(path.read_text(encoding="ascii").strip())
        return cls.from_secret(secret)

    def sign(self, board: SakkaBoard) -> str:
        return hmac.new(self.secret, board_signature_payload(board), hashlib.sha256).hexdigest()

    def verify(self, board: SakkaBoard) -> bool:
        if board.signature_algorithm != self.algorithm or board.signature_key_id != self.key_id:
            return False
        if not board.board_signature:
            return False
        return hmac.compare_digest(self.sign(board), board.board_signature)
