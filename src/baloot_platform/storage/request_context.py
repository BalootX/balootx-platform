from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class DatabaseIdentityContext:
    subject_id: str = ""
    tenant_id: str = ""
    role: str = "anonymous"
    request_id: str = ""
    session_id: str = ""
    configuration_admin: bool = False
    oidc_registry_read: bool = False


CURRENT_DATABASE_IDENTITY: ContextVar[DatabaseIdentityContext] = ContextVar(
    "baloot_database_identity", default=DatabaseIdentityContext()
)


def apply_postgres_identity_context(connection: Any, context: DatabaseIdentityContext | None = None) -> None:
    selected = context or CURRENT_DATABASE_IDENTITY.get()
    values = {
        "app.subject_id": selected.subject_id, "app.tenant_id": selected.tenant_id,
        "app.role": selected.role, "app.request_id": selected.request_id,
        "app.session_id": selected.session_id, "baloot.subject_id": selected.subject_id,
        "app.configuration_admin": "on" if selected.configuration_admin else "",
        "app.oidc_registry_read": "on" if selected.oidc_registry_read else "",
    }
    for key, value in values.items():
        connection.execute("SELECT set_config(%s,%s,true)", (key, str(value or "")))


@contextlib.contextmanager
def contextual_connection(
    pool: Any, context: DatabaseIdentityContext | None = None
) -> Iterator[Any]:
    with pool.connection() as connection:
        apply_postgres_identity_context(connection, context)
        yield connection
