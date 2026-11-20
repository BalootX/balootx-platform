from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from functools import lru_cache

from baloot_platform.identity import Identity


@lru_cache(maxsize=1)
def _derived_subject_id(configured_key: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", configured_key.encode("utf-8"),
        b"baloot-platform:api-service-subject:v2", 600_000, dklen=10,
    )
    return f"SVC-{digest.hex().upper()}"


def service_client_identity(headers: Mapping[str, str], default_name: str) -> Identity:
    """Build the synthetic identity after the transport authenticates the caller."""
    subject_id = headers.get("x-baloot-subject-id", "").strip()
    if not subject_id:
        configured_key = os.environ.get("BALOOT_API_KEY")
        subject_id = "SVC-API"
        if configured_key:
            # Deployments can pin the ID already recorded on service-owned data.
            subject_id = (
                os.environ.get("BALOOT_API_SUBJECT_ID", "").strip()
                or _derived_subject_id(configured_key)
            )
    display_name = headers.get("x-baloot-subject-name", default_name)[:40]
    return Identity(subject_id, "service", display_name, None, True, None)
