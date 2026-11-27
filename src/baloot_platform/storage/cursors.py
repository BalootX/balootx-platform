from __future__ import annotations

import base64
import json
import re
from typing import Any, Mapping


def encode_cursor_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor_payload(cursor: str, *, maximum_length: int) -> Any:
    if not isinstance(cursor, str) or not cursor or len(cursor) > maximum_length:
        raise ValueError("invalid cursor length")
    if re.fullmatch(r"[A-Za-z0-9_-]+", cursor) is None:
        raise ValueError("invalid cursor encoding")
    raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
    return json.loads(raw.decode())
