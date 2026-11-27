from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

from .models import DISPLAY_NAME_MAX_LENGTH, Identity, utc_now
from .ownership import owns_subject
from .pagination import cursor_scope, decode_cursor, encode_cursor
from .store import PostgresIdentityStore, SQLiteIdentityStore

_USERNAME = re.compile(r"^[A-Za-z0-9_\-.]{3,32}$")


def _normalize_username(username: str) -> str:
    return username.strip().lower()


class IdentityService:
    """Optional local identity suitable for guest-first deployments.

    This implementation deliberately has a replaceable boundary. The rest of
    the platform consumes only subject identities and does not depend on local
    passwords, making OIDC/passkeys a later provider swap rather than a rewrite.
    """

    def __init__(self, store: SQLiteIdentityStore | PostgresIdentityStore, *, session_hours: int = 24 * 30,
                 guest_hours: int = 24 * 7, password_minimum: int = 8):
        self.store = store
        self.session_hours = max(1, int(session_hours))
        self.guest_hours = max(1, int(guest_hours))
        self.password_minimum = max(8, int(password_minimum))

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        # OpenSSL's implicit memory cap may be too close to the 32 MiB working
        # set for n=2**15. Set an explicit bounded allowance so the same
        # password format works consistently in containers and native runs.
        return hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**15, r=8, p=1,
            dklen=32, maxmem=128 * 1024 * 1024,
        )

    @staticmethod
    def _identity(row: dict[str, Any]) -> Identity:
        return Identity(
            str(row["subject_id"]), str(row["kind"]), str(row["display_name"]),
            str(row["username"]) if row.get("username") else None,
            str(row["kind"]) != "guest", str(row.get("subject_created_at") or row.get("created_at") or ""),
        )

    def _new_session(self, subject_id: str, *, guest: bool, user_agent: str | None = None,
                     connection=None) -> dict[str, Any]:
        token = secrets.token_urlsafe(40)
        csrf = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=self.guest_hours if guest else self.session_hours)
        session_id = f"SES-{uuid.uuid4().hex.upper()}"
        row = {
            "session_id": session_id,
            "subject_id": subject_id,
            "token_hash": self._hash_token(token),
            "csrf_token": csrf,
            "created_at": now.isoformat(timespec="seconds"),
            "expires_at": expiry.isoformat(timespec="seconds"),
            "last_seen_at": now.isoformat(timespec="seconds"),
            "user_agent_hash": hashlib.sha256((user_agent or "").encode()).hexdigest() if user_agent else None,
        }
        self.store.insert_session(row, connection=connection)
        return {"token": token, "csrfToken": csrf, "sessionId": session_id, "expiresAt": row["expires_at"]}

    def create_guest(self, display_name: str | None = None, *, user_agent: str | None = None) -> dict[str, Any]:
        subject_id = f"GST-{uuid.uuid4().hex[:20].upper()}"
        name = (display_name or "ضيف البلوت").strip()[:DISPLAY_NAME_MAX_LENGTH] or "ضيف البلوت"
        created = utc_now()
        self.store.insert_subject(
            subject_id=subject_id, kind="guest", username=None, display_name=name,
            password_salt=None, password_hash=None, created_at=created,
        )
        session = self._new_session(subject_id, guest=True, user_agent=user_agent)
        identity = Identity(subject_id, "guest", name, None, False, created)
        return {"identity": identity.to_dict(), "session": session}

    def register(self, username: str, password: str, display_name: str | None = None,
                 *, user_agent: str | None = None, upgrade_guest_id: str | None = None) -> dict[str, Any]:
        username = _normalize_username(username)
        if not _USERNAME.fullmatch(username):
            raise ValueError("username must be 3-32 letters, numbers, dot, dash or underscore")
        if len(password) < self.password_minimum:
            raise ValueError(f"password must contain at least {self.password_minimum} characters")
        if self.store.get_subject_by_username(username):
            raise ValueError("username already exists")
        subject_id = upgrade_guest_id or f"USR-{uuid.uuid4().hex[:20].upper()}"
        salt = secrets.token_bytes(16)
        password_hash = self._password_hash(password, salt)
        created = utc_now()
        visible_name = (display_name or username).strip()[:DISPLAY_NAME_MAX_LENGTH] or username
        if upgrade_guest_id:
            self.store.upgrade_guest(
                subject_id=upgrade_guest_id, username=username, display_name=visible_name,
                password_salt=salt, password_hash=password_hash,
            )
        else:
            self.store.insert_subject(
                subject_id=subject_id, kind="local", username=username,
                display_name=visible_name, password_salt=salt,
                password_hash=password_hash, created_at=created,
            )
        session = self._new_session(subject_id, guest=False, user_agent=user_agent)
        return {"identity": self._identity(self.store.get_subject(subject_id) or {}).to_dict(), "session": session}

    def login(self, username: str, password: str, *, user_agent: str | None = None) -> dict[str, Any]:
        row = self.store.get_subject_by_username(_normalize_username(username))
        if not row or row.get("disabled_at") or row.get("kind") != "local":
            raise ValueError("invalid username or password")
        salt = bytes(row["password_salt"])
        expected = bytes(row["password_hash"])
        actual = self._password_hash(password, salt)
        if not hmac.compare_digest(expected, actual):
            raise ValueError("invalid username or password")
        session = self._new_session(str(row["subject_id"]), guest=False, user_agent=user_agent)
        return {"identity": self._identity(row).to_dict(), "session": session}

    def resolve_subject(self, subject_id: str) -> Identity | None:
        row = self.store.get_subject(subject_id)
        return self._identity(row) if row else None

    def resolve(self, token: str | None, *, touch: bool = True) -> tuple[Identity | None, dict[str, Any] | None]:
        if not token:
            return None, None
        row = self.store.get_session_by_hash(self._hash_token(token))
        if not row or row.get("revoked_at") or row.get("disabled_at"):
            return None, None
        try:
            expiry = datetime.fromisoformat(str(row["expires_at"]))
        except ValueError:
            return None, None
        if expiry <= datetime.now(timezone.utc):
            return None, None
        if touch:
            self.store.touch_session(str(row["session_id"]), utc_now())
        return self._identity(row), row

    def logout(self, token: str | None) -> None:
        identity, row = self.resolve(token, touch=False)
        if identity is not None and row is not None:
            self.store.revoke_session(str(row["session_id"]), utc_now())

    def create_external(self, *, kind: str, display_name: str, username: str | None = None,
                        subject_id: str | None = None, user_agent: str | None = None,
                        connection=None) -> dict[str, Any]:
        if kind not in {"oidc", "passkey", "service"}:
            raise ValueError("unsupported external identity kind")
        selected = subject_id or f"USR-{uuid.uuid4().hex[:20].upper()}"
        existing = self.store.get_subject(selected, connection=connection)
        if existing is None:
            created = utc_now()
            self.store.insert_subject(
                subject_id=selected, kind=kind, username=username,
                display_name=(display_name or "مستخدم البلوت").strip()[:DISPLAY_NAME_MAX_LENGTH] or "مستخدم البلوت",
                password_salt=None, password_hash=None, created_at=created,
                connection=connection,
            )
        row = self.store.get_subject(selected, connection=connection) or {}
        session = self._new_session(selected, guest=False, user_agent=user_agent, connection=connection)
        return {"identity": self._identity(row).to_dict(), "session": session}

    def _page(
        self,
        fetch: Callable[..., list[dict[str, Any]]],
        value: str,
        *,
        operation: str,
        keys: tuple[str, ...],
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        scope = cursor_scope(operation, value)
        rows = fetch(value, limit=limit + 1, after=decode_cursor(cursor, scope, keys=keys))
        items = rows[:limit]
        next_cursor = None
        if len(rows) > limit:
            next_cursor = encode_cursor(scope, tuple(items[-1][key] for key in keys))
        return {"items": items, "nextCursor": next_cursor}

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return [self._identity(row).to_dict() for row in self.store.search_subjects(query, limit=limit)]

    def search_page(
        self, query: str, *, limit: int, maximum_results: int, cursor: str | None = None
    ) -> dict[str, Any]:
        page = self._page(
            partial(self.store.search_subjects, exact_username=_normalize_username(query), maximum_results=maximum_results),
            query.strip(),
            operation="search", keys=("search_rank", "subject_id"), limit=limit, cursor=cursor,
        )
        page["items"] = [self._identity(row).to_dict() for row in page["items"]]
        return page

    def request_friend(self, sender_id: str, recipient_id: str) -> dict[str, Any]:
        if owns_subject(sender_id, recipient_id):
            raise ValueError("cannot add yourself")
        if self.store.get_subject(recipient_id) is None:
            raise KeyError(recipient_id)
        request_id = f"FRQ-{uuid.uuid4().hex[:20].upper()}"
        created = utc_now()
        self.store.create_friend_request(
            request_id=request_id, sender_id=sender_id, recipient_id=recipient_id, created_at=created,
        )
        return {"requestId": request_id, "senderId": sender_id, "recipientId": recipient_id, "status": "pending", "createdAt": created}

    def respond_friend(self, request_id: str, recipient_id: str, *, accept: bool) -> dict[str, Any]:
        self.store.respond_friend_request(
            request_id=request_id, recipient_id=recipient_id, accept=accept, responded_at=utc_now(),
        )
        return {"requestId": request_id, "status": "accepted" if accept else "declined"}

    def friends(self, subject_id: str) -> list[dict[str, Any]]:
        return [self._identity(row).to_dict() for row in self.store.list_friends(subject_id)]

    def requests(self, subject_id: str) -> list[dict[str, Any]]:
        return self.store.list_friend_requests(subject_id)

    def friend_page(self, subject_id: str, *, limit: int, cursor: str | None = None) -> dict[str, Any]:
        page = self._page(
            self.store.list_friends, subject_id, operation="friends",
            keys=("subject_id",), limit=limit, cursor=cursor,
        )
        page["items"] = [self._identity(row).to_dict() for row in page["items"]]
        return page

    def request_page(self, subject_id: str, *, limit: int, cursor: str | None = None) -> dict[str, Any]:
        return self._page(
            self.store.list_friend_requests, subject_id, operation="requests",
            keys=("created_at", "request_id"), limit=limit, cursor=cursor,
        )

    def friend_ids(self, subject_id: str) -> list[str]:
        return self.store.friend_ids(subject_id)
