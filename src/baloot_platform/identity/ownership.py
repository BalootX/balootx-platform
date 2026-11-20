"""Ownership lookup for the authenticated service subject compatibility scope."""

from contextvars import ContextVar

# New ID first; the legacy ID can only match an existing record in this request.
SERVICE_SUBJECT_OWNER: ContextVar[tuple[str, str] | None] = ContextVar(
    "baloot_service_subject_owner", default=None,
)


def ownership_ids(subject_id: str) -> tuple[str, str]:
    owner = SERVICE_SUBJECT_OWNER.get()
    if owner is not None and owner[0] == subject_id:
        return owner
    return subject_id, subject_id


def owns_subject(subject_id: str | None, stored_subject: str | None) -> bool:
    return subject_id is not None and stored_subject in ownership_ids(subject_id)
