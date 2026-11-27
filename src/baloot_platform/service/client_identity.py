from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from functools import lru_cache

from baloot_platform.identity import Identity
from baloot_platform.identity.ownership import SERVICE_SUBJECT_OWNER


def _derived_subject_id(configured_key: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", configured_key.encode("utf-8"),
        b"baloot-platform:api-service-subject:v2", 600_000, dklen=10,
    )
    return f"SVC-{digest.hex().upper()}"


def _legacy_subject_id(key: str) -> str:
    # CodeQL: reconstruct the already-stored public subject ID for ownership lookup only.
    # This digest is neither a credential verifier nor the ID issued to new records.
    return f"SVC-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20].upper()}"


# One configured key per process keeps the password-hard KDF off the request hot path.
@lru_cache(maxsize=1)
def _configured_subject_ownership(key: str) -> tuple[str, str]:
    return _derived_subject_id(key), _legacy_subject_id(key)


def initialize_service_subject_ownership() -> None:
    """Prepare the lookup pair before either service accepts requests."""
    configured_key = os.environ.get("BALOOT_API_KEY")
    if configured_key:
        _configured_subject_ownership(configured_key)


def service_subject_ownership(
    headers: Mapping[str, str], *, configured_key: str | None,
) -> tuple[str, str] | None:
    """Resolve the service subject's existing ownership after authentication."""
    if headers.get("x-baloot-subject-id", "").strip():
        return None
    if configured_key:
        return _configured_subject_ownership(configured_key)
    return None


def service_client_identity(headers: Mapping[str, str], default_name: str) -> Identity:
    """Build the synthetic identity after the transport authenticates the caller."""
    subject_id = headers.get("x-baloot-subject-id", "").strip()
    if not subject_id:
        ownership = SERVICE_SUBJECT_OWNER.get()
        if ownership is not None:
            subject_id = ownership[0]
        else:
            configured_key = os.environ.get("BALOOT_API_KEY")
            subject_id = _configured_subject_ownership(configured_key)[0] if configured_key else "SVC-API"
    display_name = headers.get("x-baloot-subject-name", default_name)[:40]
    return Identity(subject_id, "service", display_name, None, True, None)
