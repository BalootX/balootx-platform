from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable
from pathlib import Path


_SQL_TOKEN = re.compile(
    r"""(?P<line>--[^\r\n]*)|(?P<block>/\*)|(?P<dollar>\$(?:[^\W\d]\w*)?\$)"""
    r"""|[Ee]'(?:\\.|''|[^'\\])*'|'(?:''|[^'])*'|"(?:""|[^"])*"|[\w$]+|\S""",
    re.DOTALL,
)
_COMMENT_END = re.compile(r"/\*|\*/")
_CONCURRENT_INDEX_PREFIX = "CREATE INDEX CONCURRENTLY "
_CONCURRENT_INDEX = re.compile(
    r"CREATE INDEX CONCURRENTLY IF NOT EXISTS ([a-z][a-z0-9_]*)\s+ON ([a-z][a-z0-9_]*)\."
)
_TRANSACTION_COMMANDS = {"ABORT", "BEGIN", "COMMIT", "END", "RELEASE", "ROLLBACK", "SAVEPOINT", "START"}
_ROUTINE_HEADER = {
    ("CREATE", "OR"): "CREATE OR",
    ("CREATE OR", "REPLACE"): "CREATE OR REPLACE",
    ("CREATE", "FUNCTION"): "ROUTINE",
    ("CREATE", "PROCEDURE"): "ROUTINE",
    ("CREATE OR REPLACE", "FUNCTION"): "ROUTINE",
    ("CREATE OR REPLACE", "PROCEDURE"): "ROUTINE",
}
_ATOMIC_MIGRATION_GROUPS = (
    (
        "030_oidc_issuer_transition.sql",
        "031_oidc_registry_access.sql",
        "032_oidc_identity_authority.sql",
        "033_oidc_database_roles.sql",
    ),
)


def migration_batches(names: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    batches = {name: (name,) for name in sorted(names)}
    for group in _ATOMIC_MIGRATION_GROUPS:
        present = [name for name in group if name in batches]
        if not present:
            continue
        if len(present) != len(group):
            raise ValueError(f"Incomplete PostgreSQL migration batch: {group[0]} through {group[-1]}")
        for name in group[:-1]:
            del batches[name]
        # Run independent interleaved concurrent indexes before opening the batch transaction.
        batches[group[-1]] = group
    return tuple(batches.values())


def _migration_sql(source: bytes, name: str) -> str:
    sql = source.decode("utf-8").strip()
    # Preserve the old files and commit their SQL and receipts together.
    if sql.startswith("BEGIN;") and sql.endswith("COMMIT;"):
        sql = sql.removeprefix("BEGIN;").removesuffix("COMMIT;")

    position = 0
    command = None
    atomic_depth = 0
    while token := _SQL_TOKEN.search(sql, position):
        position = token.end()
        if token.lastgroup == "line":
            continue
        if token.lastgroup == "block":
            depth = 1
            while depth:
                marker = _COMMENT_END.search(sql, position)
                if marker is None:
                    raise ValueError(f"Unterminated SQL comment in {name}")
                position = marker.end()
                depth += 1 if marker[0] == "/*" else -1
            continue
        if token.lastgroup == "dollar":
            closing = sql.find(token[0], position)
            if closing < 0:
                raise ValueError(f"Unterminated SQL dollar quote in {name}")
            position = closing + len(token[0])
        word = token[0].upper()
        if word == ";":
            command = None
        elif command is None:
            # A routine's END follows a complete statement. CASE ... END is part of one.
            if atomic_depth and word == "END":
                atomic_depth -= 1
            elif not atomic_depth and word in _TRANSACTION_COMMANDS:
                raise ValueError(f"Unexpected transaction control in {name}: {word}")
            command = word
        elif command == "PREPARE":
            if not atomic_depth and word == "TRANSACTION":
                raise ValueError(f"Unexpected transaction control in {name}: PREPARE TRANSACTION")
            command = ""
        elif command in ("CREATE", "CREATE OR", "CREATE OR REPLACE"):
            command = _ROUTINE_HEADER.get((command, word), "")
        elif command in ("ROUTINE", "ROUTINE BEGIN"):
            if command == "ROUTINE BEGIN" and word == "ATOMIC":
                atomic_depth += 1
                command = None
            else:
                command = "ROUTINE BEGIN" if word == "BEGIN" else "ROUTINE"
    if atomic_depth:
        raise ValueError(f"Unterminated SQL BEGIN ATOMIC body in {name}")
    return sql


def _apply_concurrent_index(conn, sql: str, name: str, *, recorded: bool) -> bool:
    match = _CONCURRENT_INDEX.match(sql)
    if match is None or sql.count(";") != 1 or not sql.endswith(";"):
        raise ValueError(f"{name} must contain one schema-qualified concurrent index build")
    index, schema = match.groups()
    qualified = f"{schema}.{index}"
    # The predecessor runner shares this index lock during a service rollback.
    conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (qualified,))
    try:
        valid = conn.execute(
            "SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(%s)", (qualified,)
        ).fetchone()
        if recorded and valid == (True,):
            return False
        # An interrupted build can leave an invalid index that IF NOT EXISTS would skip.
        if valid == (False,):
            conn.execute(f"DROP INDEX CONCURRENTLY {qualified}")
        conn.execute(sql)
        return True
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (qualified,))


def apply_migrations(dsn: str, directory: Path) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(hashtext('baloot.schema-migrations'))")
        conn.execute("CREATE SCHEMA IF NOT EXISTS baloot_migrations")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS baloot_migrations.applied (
                name text PRIMARY KEY,
                sha256 text NOT NULL
            )"""
        )
        applied = dict(conn.execute("SELECT name, sha256 FROM baloot_migrations.applied"))
        migrations = {}
        for migration in sorted(directory.glob("*.sql")):
            source = migration.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if migration.name in applied and applied[migration.name] != digest:
                raise ValueError(f"Applied migration has changed: {migration.name}")
            migrations[migration.name] = (digest, _migration_sql(source, migration.name))
        batches = migration_batches(migrations)
        for batch in batches:
            if len(batch) > 1:
                recorded = tuple(name for name in batch if name in applied)
                if recorded != batch[:len(recorded)]:
                    raise ValueError(f"Non-prefix receipts for PostgreSQL migration batch: {batch[0]}")
                if any(migrations[name][1].startswith(_CONCURRENT_INDEX_PREFIX) for name in batch):
                    raise ValueError(f"Concurrent index in atomic PostgreSQL migration batch: {batch[0]}")

        repaired = set()
        for batch in batches:
            first = batch[0]
            concurrent = migrations[first][1].startswith(_CONCURRENT_INDEX_PREFIX)
            if concurrent and _apply_concurrent_index(
                conn, migrations[first][1], first, recorded=first in applied,
            ):
                repaired.add(first)
            pending = [name for name in batch if name not in applied]
            if pending:
                with conn.transaction():
                    for name in pending:
                        digest, sql = migrations[name]
                        if not concurrent:
                            conn.execute(sql)
                        conn.execute(
                            "INSERT INTO baloot_migrations.applied (name, sha256) VALUES (%s,%s)",
                            (name, digest),
                        )

        # Inventory output remains sorted by full filename for migration receipts and proofs.
        for name in migrations:
            action = "applied" if name not in applied else "repaired" if name in repaired else "verified"
            print(f"{action} {name}")


def main() -> None:
    dsn = os.environ.get("BALOOT_DATABASE_URL", "").strip()
    if not dsn:
        raise SystemExit("BALOOT_DATABASE_URL is required")
    root = Path(os.environ.get("BALOOT_ROOT", Path(__file__).resolve().parents[1]))
    try:
        apply_migrations(dsn, root / "migrations")
    except ImportError as exc:
        raise SystemExit("install baloot-platform[postgres]") from exc
    print("Baloot PostgreSQL schema is current")


if __name__ == "__main__":
    main()
