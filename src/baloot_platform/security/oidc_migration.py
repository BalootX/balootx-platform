from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import httpx

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.configuration import PostgresConfigurationStore, VersionedConfigurationService
from baloot_platform.storage.request_context import (
    CURRENT_DATABASE_IDENTITY,
    DatabaseIdentityContext,
    apply_postgres_identity_context,
)

from .oidc import (
    DEFAULT_OIDC_HTTP_TIMEOUT,
    LEGACY_DEFAULT_PROVIDER_ID,
    OIDC_PROFILE_ENV,
    OIDC_PROVIDER_FAMILY,
    discover_oidc_issuer,
)
from .state_v20 import oidc_identity_rls_active


def legacy_oidc_profile(
    client: httpx.Client, *, profile_id: str, version: str, reason: str, algorithms: list[str]
) -> tuple[str, dict[str, Any]]:
    fields = ("issuer", "authorization_endpoint", "token_endpoint", "client_id", "redirect_uri")
    provider = {field: os.environ.get(f"BALOOT_OIDC_{field.upper()}") for field in fields}
    if any(not value for value in provider.values()):
        raise ValueError("complete legacy OIDC provider metadata is required")
    legacy_issuer = provider["issuer"]
    assert legacy_issuer is not None
    if "BALOOT_OIDC_JWKS_URI" in os.environ:
        logging.getLogger(__name__).warning("The legacy JWKS override is ignored; OIDC Discovery is required")
    issuer, _ = discover_oidc_issuer(client, legacy_issuer, timeout=DEFAULT_OIDC_HTTP_TIMEOUT)
    if issuer.rstrip("/") != legacy_issuer.rstrip("/"):
        raise ValueError("discovered issuer differs from legacy issuer beyond a trailing slash")
    provider["issuer"] = issuer
    client_secret = os.environ.get("BALOOT_OIDC_CLIENT_SECRET")
    if client_secret and not client_secret.strip():
        raise ValueError("configured legacy OIDC client secret is blank")
    if client_secret:
        provider["client_secret_env"] = "BALOOT_OIDC_CLIENT_SECRET"
    return legacy_issuer, {
        "schema_version": 1,
        "profile": {"id": profile_id, "version": version},
        "change_reason": reason,
        "providers": {
            os.environ.get("BALOOT_OIDC_PROVIDER_ID", LEGACY_DEFAULT_PROVIDER_ID): {
                **provider, "allowed_algorithms": algorithms,
            },
        },
    }


def _lock_migration(conn: Any) -> None:
    conn.execute(
        "LOCK TABLE configuration.profile_revisions, security.oidc_issuer_transitions, "
        "security.external_identities IN SHARE ROW EXCLUSIVE MODE"
    )


def migrate_legacy_oidc(
    conn: Any, configuration: VersionedConfigurationService, *,
    payload: dict[str, Any], legacy_issuer: str, actor: str,
) -> dict[str, Any]:
    payload = configuration.validate_profile(OIDC_PROVIDER_FAMILY, payload)
    [(provider_id, provider)] = payload["providers"].items()
    profile_id = payload["profile"]["id"]
    issuer = provider["issuer"]
    if issuer.rstrip("/") != legacy_issuer.rstrip("/"):
        raise ValueError("issuer migration must preserve the legacy issuer identity")
    _lock_migration(conn)
    prior = conn.execute(
        "SELECT * FROM security.oidc_issuer_transitions WHERE rolled_back_at IS NULL "
        "AND (provider_id = %s OR profile_id = %s)",
        (provider_id, profile_id),
    ).fetchall()
    if prior:
        if len(prior) != 1:
            raise ValueError("provider and profile are bound to different migrations")
        transition = prior[0]
        revision = configuration.store.get_profile(transition["revision_id"])
        if (
            revision is None
            or revision.payload != {**payload, "change_reason": revision.payload["change_reason"]}
            or transition["legacy_issuer"] != legacy_issuer
        ):
            raise ValueError("existing OIDC migration has a different binding")
        return {"revisionId": revision.revision_id, "status": "already-applied"}
    try:
        configuration.resolve_profile(family=OIDC_PROVIDER_FAMILY, profile_id=profile_id)
    except KeyError:
        pass
    else:
        raise ValueError("OIDC profile already has a published revision")

    if issuer != legacy_issuer and conn.execute(
        "SELECT 1 FROM security.external_identities WHERE provider_id = %s AND issuer = %s LIMIT 1",
        (provider_id, issuer),
    ).fetchone():
        raise ValueError("discovered issuer already has identity links for this provider")

    revision = configuration.create_profile(
        family=OIDC_PROVIDER_FAMILY, payload=payload, created_by=actor, publish=True
    )
    revision_id = revision["revisionId"]
    if revision["status"] == "draft":
        revision = configuration.publish_profile(
            revision_id, expected_etag=revision["etag"], actor_id=actor
        )
    if revision["status"] != "published":
        raise ValueError("migration requires a new profile version")
    conn.execute(
        "INSERT INTO security.oidc_issuer_transitions "
        "(revision_id, profile_id, provider_id, legacy_issuer, issuer) VALUES (%s,%s,%s,%s,%s)",
        (revision_id, profile_id, provider_id, legacy_issuer, issuer),
    )
    count = 0
    if issuer != legacy_issuer:
        count = conn.execute(
            "UPDATE security.external_identities SET issuer = %s WHERE provider_id = %s AND issuer = %s",
            (issuer, provider_id, legacy_issuer),
        ).rowcount
        conn.execute(
            "INSERT INTO security.external_identities "
            "(provider_id, issuer, external_subject, local_subject_id, claims_json, linked_at) "
            "SELECT provider_id, %s, external_subject, local_subject_id, claims_json, linked_at "
            "FROM security.external_identities WHERE provider_id = %s AND issuer = %s",
            (legacy_issuer, provider_id, issuer),
        )
    configuration.store.append_event(
        "profile", revision_id, "oidc_issuer_migrated", actor,
        {"providerId": provider_id, "legacyIssuer": legacy_issuer, "issuer": issuer, "links": count},
    )
    return {"revisionId": revision_id, "status": "applied", "links": count}


def _migration_transition(conn: Any, profile_id: str, revision_id: str) -> dict[str, Any]:
    _lock_migration(conn)
    transition = conn.execute(
        "SELECT * FROM security.oidc_issuer_transitions WHERE profile_id = %s AND revision_id = %s",
        (profile_id, revision_id),
    ).fetchone()
    if transition is None:
        raise ValueError("OIDC migration was not found for the selected profile and revision")
    return transition


def _delete_aliases(conn: Any, transition: dict[str, Any]) -> int:
    return conn.execute(
        "DELETE FROM security.external_identities WHERE provider_id = %s AND issuer = %s AND issuer_alias",
        (transition["provider_id"], transition["legacy_issuer"]),
    ).rowcount


def rollback_legacy_oidc(
    conn: Any, configuration: VersionedConfigurationService, *,
    profile_id: str, revision_id: str, actor: str, reason: str,
) -> dict[str, Any]:
    transition = _migration_transition(conn, profile_id, revision_id)
    if transition["rolled_back_at"] is not None:
        return {"revisionId": revision_id, "status": "already-rolled-back"}
    if transition["finalized_at"] is not None:
        raise ValueError("the OIDC migration rollback window has closed")
    try:
        active = configuration.resolve_profile(family=OIDC_PROVIDER_FAMILY, profile_id=profile_id)
    except KeyError:
        active = None

    conn.execute(
        "UPDATE security.oidc_issuer_transitions SET rolled_back_at = now() WHERE revision_id = %s",
        (revision_id,),
    )
    provider_id, issuer, legacy_issuer = (
        transition["provider_id"], transition["issuer"], transition["legacy_issuer"],
    )
    count = 0
    if issuer != legacy_issuer:
        _delete_aliases(conn, transition)
        count = conn.execute(
            "UPDATE security.external_identities SET issuer = %s WHERE provider_id = %s AND issuer = %s",
            (legacy_issuer, provider_id, issuer),
        ).rowcount
    if active is not None:
        configuration.retire_profile(
            active["revisionId"], expected_etag=active["etag"], actor_id=actor, reason=reason
        )
    configuration.store.append_event(
        "profile", revision_id, "oidc_issuer_rolled_back", actor,
        {"links": count, "retiredRevisionId": active["revisionId"] if active else None, "reason": reason},
    )
    return {"revisionId": revision_id, "status": "rolled-back", "links": count}


def finalize_legacy_oidc(
    conn: Any, configuration: VersionedConfigurationService, *,
    profile_id: str, revision_id: str, actor: str, reason: str,
) -> dict[str, Any]:
    transition = _migration_transition(conn, profile_id, revision_id)
    if transition["rolled_back_at"] is not None:
        raise ValueError("the OIDC migration was rolled back")
    if transition["finalized_at"] is not None:
        return {"revisionId": revision_id, "status": "already-finalized"}
    conn.execute(
        "UPDATE security.oidc_issuer_transitions SET finalized_at = now() WHERE revision_id = %s",
        (revision_id,),
    )
    count = _delete_aliases(conn, transition)
    configuration.store.append_event(
        "profile", revision_id, "oidc_issuer_migration_finalized", actor,
        {"aliasesRemoved": count, "reason": reason},
    )
    return {"revisionId": revision_id, "status": "finalized", "aliasesRemoved": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate or roll back a legacy OIDC provider")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--version")
    parser.add_argument("--algorithm", action="append", default=[])
    transition = parser.add_mutually_exclusive_group()
    transition.add_argument("--rollback", metavar="REVISION_ID")
    transition.add_argument("--finalize", metavar="REVISION_ID")
    args = parser.parse_args()
    profile_id = os.environ.get(OIDC_PROFILE_ENV, "").strip()
    dsn = os.environ.get("BALOOT_DATABASE_URL", "").strip()
    actor, reason = args.actor.strip(), args.reason.strip()
    if not (dsn and profile_id and actor and reason):
        parser.error(f"BALOOT_DATABASE_URL, {OIDC_PROFILE_ENV}, actor and reason are required")
    closing_revision = args.rollback or args.finalize
    if not closing_revision and not (args.version and args.algorithm):
        parser.error("forward migration requires --version and at least one --algorithm")

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        raise SystemExit('Install with: pip install "baloot-platform[postgres]"') from None

    try:
        if not closing_revision:
            with httpx.Client(follow_redirects=False) as client:
                legacy_issuer, payload = legacy_oidc_profile(
                    client, profile_id=profile_id, version=args.version,
                    reason=reason, algorithms=args.algorithm,
                )
        root = platform_root()
        control_plane = build_bundle(root).component("configuration")
        context = DatabaseIdentityContext(
            subject_id=actor, role="service", configuration_admin=True, oidc_identity_authority="migration",
        )
        identity_token = CURRENT_DATABASE_IDENTITY.set(context)
        try:
            with psycopg.connect(dsn, row_factory=dict_row) as conn:
                access = conn.execute(
                    "SELECT row_security_active('configuration.profile_revisions') "
                    "AND has_table_privilege(session_user, 'security.oidc_issuer_transitions', 'UPDATE') AS allowed"
                ).fetchone()
                if not access["allowed"] or not oidc_identity_rls_active(conn):
                    raise RuntimeError(
                        "OIDC issuer migration requires a dedicated login subject to row-level security "
                        "and the current identity policies"
                    )
                apply_postgres_identity_context(conn, context)
                store = PostgresConfigurationStore(dsn, transaction_connection=conn)
                configuration = VersionedConfigurationService(store, root=root, profile=control_plane)
                if closing_revision:
                    close_migration = rollback_legacy_oidc if args.rollback else finalize_legacy_oidc
                    result = close_migration(
                        conn, configuration, profile_id=profile_id, revision_id=closing_revision,
                        actor=actor, reason=reason,
                    )
                else:
                    result = migrate_legacy_oidc(
                        conn, configuration, payload=payload, legacy_issuer=legacy_issuer, actor=actor
                    )
        finally:
            CURRENT_DATABASE_IDENTITY.reset(identity_token)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    except psycopg.Error as exc:
        # Database errors can include account identifiers and connection details.
        raise SystemExit(f"OIDC database transaction failed ({exc.sqlstate or 'connection error'})") from None
    print(json.dumps(result, sort_keys=True))
