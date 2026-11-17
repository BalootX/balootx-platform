"""Safe, versioned, deterministic serialization for Baloot platform records.

The V7 storage codec replaces Python pickle in durable stores.  It serializes
only Baloot-platform dataclasses, enums and JSON-compatible primitives.  The
canonical byte representation is stable across processes and auditable.
"""
from __future__ import annotations

import base64
import importlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

MAGIC = b"BALOOT7\x00"
SCHEMA_VERSION = 1
_ALLOWED_PREFIXES = (
    "baloot_platform.core.",
    "baloot_platform.analysis.",
    "baloot_platform.audit.",
    "baloot_platform.challenge.",
    "baloot_platform.service.game",
    "baloot_platform.learning.",
    "baloot_platform.duplicate.",
    "baloot_platform.history.",
)


def _qualified(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _resolve(name: str) -> type[Any]:
    if not name.startswith(_ALLOWED_PREFIXES):
        raise ValueError(f"durable payload type is not allow-listed: {name}")
    module_name, _, qualname = name.rpartition(".")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in qualname.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise TypeError(f"durable payload target is not a type: {name}")
    return value


def to_wire(value: Any) -> Any:
    # IntEnum is also an int, so enum detection must happen first.
    if isinstance(value, Enum):
        return {"$enum": _qualified(type(value)), "value": value.value}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, tuple):
        return {"$tuple": [to_wire(item) for item in value]}
    if isinstance(value, set):
        encoded = [to_wire(item) for item in value]
        encoded.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"$set": encoded}
    if isinstance(value, list):
        return [to_wire(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: to_wire(item) for key, item in sorted(value.items())}
        pairs = [(to_wire(key), to_wire(item)) for key, item in value.items()]
        pairs.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
        return {"$map": pairs}
    if is_dataclass(value):
        if _qualified(type(value)) == "baloot_platform.service.game.GameSession":
            return {
                "$session": {
                    field.name: to_wire(getattr(value, field.name))
                    for field in fields(value)
                    if field.name != "engine"
                }
            }
        return {
            "$dataclass": _qualified(type(value)),
            "fields": {field.name: to_wire(getattr(value, field.name)) for field in fields(value)},
        }
    raise TypeError(f"unsupported durable payload type: {type(value)!r}")


def from_wire(
    value: Any,
    *,
    root: str | Path | None = None,
    bundle_provider: Callable[[str], Any] | None = None,
) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [from_wire(item, root=root, bundle_provider=bundle_provider) for item in value]
    if not isinstance(value, dict):
        raise TypeError(f"invalid durable payload node: {type(value)!r}")
    if "$enum" in value:
        cls = _resolve(str(value["$enum"]))
        return cls(value["value"])
    if "$bytes" in value:
        return base64.b64decode(str(value["$bytes"]).encode("ascii"), validate=True)
    if "$path" in value:
        return Path(str(value["$path"]))
    if "$tuple" in value:
        return tuple(from_wire(item, root=root, bundle_provider=bundle_provider) for item in value["$tuple"])
    if "$set" in value:
        return set(from_wire(item, root=root, bundle_provider=bundle_provider) for item in value["$set"])
    if "$map" in value:
        return {
            from_wire(key, root=root, bundle_provider=bundle_provider):
            from_wire(item, root=root, bundle_provider=bundle_provider)
            for key, item in value["$map"]
        }
    if "$dataclass" in value:
        cls = _resolve(str(value["$dataclass"]))
        kwargs = {
            name: from_wire(item, root=root, bundle_provider=bundle_provider)
            for name, item in value.get("fields", {}).items()
        }
        return cls(**kwargs)
    if "$session" in value:
        from baloot_platform.config import build_bundle, platform_root
        from baloot_platform.core.engine import BalootEngine
        from baloot_platform.service.game import GameSession

        kwargs = {
            name: from_wire(item, root=root, bundle_provider=bundle_provider)
            for name, item in value["$session"].items()
        }
        resolved_root = platform_root(root)
        mode = str(kwargs.get("mode", "full"))
        bundle = bundle_provider(mode) if bundle_provider is not None else build_bundle(resolved_root, mode=mode)
        if kwargs.get("bundle_hash") and kwargs["bundle_hash"] != bundle.sha256:
            raise RuntimeError(
                "stored game bundle differs from the configured immutable bundle; "
                "install the referenced profile bundle before reopening"
            )
        kwargs["engine"] = BalootEngine(bundle.rules)
        return GameSession(**kwargs)
    return {
        str(key): from_wire(item, root=root, bundle_provider=bundle_provider)
        for key, item in value.items()
    }


def encode(value: Any) -> bytes:
    envelope = {"schema": "baloot.durable", "version": SCHEMA_VERSION, "value": to_wire(value)}
    body = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + body


def decode(
    payload: bytes | bytearray | memoryview,
    *,
    root: str | Path | None = None,
    bundle_provider: Callable[[str], Any] | None = None,
) -> Any:
    raw = bytes(payload)
    if not raw.startswith(MAGIC):
        raise ValueError("unsupported durable payload format; pickle migration is required")
    envelope = json.loads(raw[len(MAGIC):].decode("utf-8"))
    if envelope.get("schema") != "baloot.durable" or int(envelope.get("version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported durable payload schema version")
    return from_wire(envelope["value"], root=root, bundle_provider=bundle_provider)


def canonical_json(value: Any) -> str:
    return json.dumps(to_wire(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
