from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .result_library import normalize_timestamp
from .serialization import encode_json

RecipeMutationKind = Literal["create", "retire"]
RecipeLifecycleStatus = Literal["draft", "published", "retired"]

IDEMPOTENCY_KEY_MAX_LENGTH = 200
RECIPE_LIST_CURSOR_MAX_LENGTH = 500
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"

_IDEMPOTENCY_KEY_RE = re.compile(IDEMPOTENCY_KEY_PATTERN)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECIPE_CODE_RE = re.compile(r"RCP-[A-Z0-9]{6,20}")
_CURSOR_KEYS = frozenset(
    {
        "schemaVersion",
        "kind",
        "order",
        "createdAt",
        "recipeCode",
        "status",
    }
)
_CURSOR_KIND = "research_recipe_list"
_CURSOR_ORDER = "created_at_desc_recipe_code_asc"


class ResearchMutationConflictError(ValueError):
    """An idempotency key was reused for a different mutation intent."""


class ResearchMutationReceiptCorruptionError(RuntimeError):
    """A durable mutation receipt cannot be replayed safely."""


def validate_idempotency_key(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Idempotency-Key must be a string")
    key = value
    if not _IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise ValueError(
            "Idempotency-Key must be 1-200 ASCII letters, digits, '.', '_', ':', or '-'"
        )
    return key


def recipe_mutation_intent_hash(
    mutation_kind: RecipeMutationKind,
    intent: Mapping[str, Any],
) -> str:
    core = {
        "schemaVersion": 1,
        "mutationKind": mutation_kind,
        "intent": copy.deepcopy(dict(intent)),
    }
    return hashlib.sha256(encode_json(core).encode("utf-8")).hexdigest()


def validate_request_intent_hash(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("research recipe mutation intent hash must be lowercase SHA-256")
    return value


def replay_recipe_mutation_receipt(
    receipt: Mapping[str, Any],
    *,
    mutation_kind: RecipeMutationKind,
    request_intent_hash: str,
) -> dict[str, Any]:
    value = dict(receipt)
    if (
        value.get("mutationKind") != mutation_kind
        or value.get("requestIntentHash") != request_intent_hash
    ):
        raise ResearchMutationConflictError(
            "Idempotency-Key was already used with a different research recipe mutation intent"
        )
    expected_status = 201 if mutation_kind == "create" else 200
    if value.get("responseStatus") != expected_status:
        raise ResearchMutationReceiptCorruptionError(
            "stored research recipe mutation receipt is invalid"
        )
    response = value.get("response")
    if not isinstance(response, Mapping):
        raise ResearchMutationReceiptCorruptionError(
            "stored research recipe mutation receipt is invalid"
        )
    return copy.deepcopy(dict(response))


@dataclass(frozen=True, slots=True)
class RecipeListCursorV1:
    created_at: str
    recipe_code: str
    status: RecipeLifecycleStatus | None

    @classmethod
    def from_anchor(
        cls,
        created_at: Any,
        recipe_code: Any,
        *,
        status: RecipeLifecycleStatus | None,
    ) -> "RecipeListCursorV1":
        if status not in {None, "draft", "published", "retired"}:
            raise ValueError("research recipe status scope is invalid")
        if not isinstance(recipe_code, str) or _RECIPE_CODE_RE.fullmatch(recipe_code) is None:
            raise ValueError("research recipe cursor anchor is invalid")
        try:
            normalized_created_at = normalize_timestamp(created_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("research recipe cursor anchor is invalid") from exc
        if normalized_created_at is None:
            raise ValueError("research recipe cursor anchor is invalid")
        return cls(normalized_created_at, recipe_code, status)

    def encode(self) -> str:
        payload = {
            "schemaVersion": 1,
            "kind": _CURSOR_KIND,
            "order": _CURSOR_ORDER,
            "createdAt": self.created_at,
            "recipeCode": self.recipe_code,
            "status": self.status,
        }
        raw = encode_json(payload).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def decode(
        cls,
        value: str,
        *,
        status: RecipeLifecycleStatus | None,
    ) -> "RecipeListCursorV1":
        if not isinstance(value, str):
            raise ValueError("research recipe cursor is malformed")
        cursor = value
        if not cursor or len(cursor) > RECIPE_LIST_CURSOR_MAX_LENGTH:
            raise ValueError("research recipe cursor is malformed")
        if re.fullmatch(r"[A-Za-z0-9_-]+", cursor) is None:
            raise ValueError("research recipe cursor is malformed")
        try:
            raw = base64.b64decode(
                cursor + "=" * (-len(cursor) % 4),
                altchars=b"-_",
                validate=True,
            )
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("research recipe cursor is malformed") from exc
        if not isinstance(payload, dict) or set(payload) != _CURSOR_KEYS:
            raise ValueError("research recipe cursor is malformed")
        if (
            type(payload.get("schemaVersion")) is not int
            or payload.get("schemaVersion") != 1
        ):
            raise ValueError("unsupported research recipe cursor version")
        if (
            payload.get("kind") != _CURSOR_KIND
            or payload.get("order") != _CURSOR_ORDER
        ):
            raise ValueError("research recipe cursor is malformed")
        if payload.get("status") != status:
            raise ValueError("research recipe cursor does not match the status scope")
        try:
            decoded = cls.from_anchor(
                payload.get("createdAt"),
                payload.get("recipeCode"),
                status=status,
            )
        except ValueError as exc:
            raise ValueError("research recipe cursor is malformed") from exc
        if payload.get("createdAt") != decoded.created_at:
            raise ValueError("research recipe cursor is malformed")
        if decoded.encode() != cursor:
            raise ValueError("research recipe cursor is malformed")
        return decoded


__all__ = [
    "IDEMPOTENCY_KEY_MAX_LENGTH",
    "IDEMPOTENCY_KEY_PATTERN",
    "RECIPE_LIST_CURSOR_MAX_LENGTH",
    "RecipeLifecycleStatus",
    "RecipeListCursorV1",
    "RecipeMutationKind",
    "ResearchMutationConflictError",
    "ResearchMutationReceiptCorruptionError",
    "recipe_mutation_intent_hash",
    "replay_recipe_mutation_receipt",
    "validate_idempotency_key",
    "validate_request_intent_hash",
]
