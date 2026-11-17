from __future__ import annotations

import json
from typing import Any, Mapping


def encode_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_json(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else json.loads(str(value))
