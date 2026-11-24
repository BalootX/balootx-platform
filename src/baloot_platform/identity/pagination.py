from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CURSOR_MAX_LENGTH = 2048


class IdentityPaginationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_page_size: int = Field(gt=0, strict=True)
    maximum_page_size: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_page_sizes(self) -> Self:
        if self.default_page_size > self.maximum_page_size:
            raise ValueError("default_page_size must not exceed maximum_page_size")
        return self

    @classmethod
    def load(cls, root: Path) -> Self:
        path = root / "configs/identity/pagination_v1.yaml"
        try:
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (ValidationError, yaml.YAMLError) as exc:
            raise ValueError(f"invalid pagination_v1.yaml: {exc}") from exc


class InvalidIdentityCursor(ValueError):
    pass


def cursor_scope(operation: str, value: str) -> str:
    return hashlib.sha256(json.dumps([1, operation, value]).encode()).hexdigest()


def encode_cursor(scope: str, anchor: tuple[str, ...]) -> str:
    raw = json.dumps([scope, *anchor], separators=(",", ":"), ensure_ascii=False).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(
    value: str | None, scope: str, *, keys: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        if not value or len(value) > CURSOR_MAX_LENGTH:
            raise ValueError
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode())
        if not isinstance(payload, list) or len(payload) != len(keys) + 1 or payload[0] != scope:
            raise ValueError
        anchor = tuple(payload[1:])
        if any(not isinstance(part, str) or "\0" in part for part in anchor):
            raise ValueError
        if "created_at" in keys:
            timestamp = anchor[keys.index("created_at")]
            parsed = datetime.fromisoformat(timestamp)
            if parsed.utcoffset() is None or parsed.astimezone(timezone.utc).isoformat() != timestamp:
                raise ValueError
        if encode_cursor(scope, anchor) != value:
            raise ValueError
        return anchor
    except (ValueError, OverflowError, RecursionError) as exc:
        raise InvalidIdentityCursor("invalid cursor for this list") from exc
