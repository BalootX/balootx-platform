from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DISPLAY_NAME_MAX_LENGTH = 40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Identity:
    subject_id: str
    kind: str
    display_name: str
    username: str | None = None
    authenticated: bool = False
    created_at: str | None = None

    @property
    def is_guest(self) -> bool:
        return self.kind == "guest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjectId": self.subject_id,
            "kind": self.kind,
            "displayName": self.display_name,
            "username": self.username,
            "authenticated": self.authenticated,
            "guest": self.is_guest,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    subject_id: str
    token_hash: str
    csrf_token: str
    created_at: str
    expires_at: str
    last_seen_at: str
    user_agent_hash: str | None = None
    revoked_at: str | None = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None
