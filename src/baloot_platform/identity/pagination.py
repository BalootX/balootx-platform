from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from baloot_platform.storage.cursors import decode_cursor_payload, encode_cursor_payload

CURSOR_MAX_LENGTH = 2048


class IdentityPaginationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_page_size: int = Field(gt=0, strict=True)
    maximum_page_size: int = Field(gt=0, strict=True)
    minimum_search_query_length: int = Field(gt=0, strict=True)
    maximum_search_query_length: int = Field(gt=0, strict=True)
    maximum_search_results: int = Field(gt=0, strict=True)
    allow_guest_search: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.default_page_size > self.maximum_page_size:
            raise ValueError("default_page_size must not exceed maximum_page_size")
        if self.minimum_search_query_length > self.maximum_search_query_length:
            raise ValueError("minimum_search_query_length must not exceed maximum_search_query_length")
        return self


class InvalidIdentityCursor(ValueError):
    pass


def cursor_scope(operation: str, value: str) -> str:
    return hashlib.sha256(json.dumps([1, operation, value]).encode()).hexdigest()


def encode_cursor(scope: str, anchor: tuple[str, ...]) -> str:
    return encode_cursor_payload({"scope": scope, "anchor": list(anchor)})


def decode_cursor(
    value: str | None, scope: str, *, keys: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        payload = decode_cursor_payload(value, maximum_length=CURSOR_MAX_LENGTH)
        if not isinstance(payload, dict) or payload.get("scope") != scope:
            raise ValueError
        if not isinstance(parts := payload.get("anchor"), list) or len(parts) != len(keys):
            raise ValueError
        anchor = tuple(parts)
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
