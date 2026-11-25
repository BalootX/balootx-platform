from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


_SQL_TOKEN = re.compile(
    r"""(?P<line>--[^\r\n]*)|(?P<block>/\*)|(?P<dollar>\$(?:[^\W\d]\w*)?\$)"""
    r"""|[Ee]'(?:\\.|''|[^'\\])*'|'(?:''|[^'])*'|"(?:""|[^"])*"|[\w$]+|\S""",
    re.DOTALL,
)
_COMMENT_END = re.compile(r"/\*|\*/")
_TRANSACTION_COMMANDS = {"ABORT", "BEGIN", "COMMIT", "END", "RELEASE", "ROLLBACK", "SAVEPOINT", "START"}
_ROUTINE_HEADER = {
    ("CREATE", "OR"): "CREATE OR",
    ("CREATE OR", "REPLACE"): "CREATE OR REPLACE",
    ("CREATE", "FUNCTION"): "ROUTINE",
    ("CREATE", "PROCEDURE"): "ROUTINE",
    ("CREATE OR REPLACE", "FUNCTION"): "ROUTINE",
    ("CREATE OR REPLACE", "PROCEDURE"): "ROUTINE",
}


def _migration_sql(source: bytes, name: str) -> str:
    sql = source.decode("utf-8").strip()
    # Preserve the old files and commit each migration with its receipt.
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
        for migration in sorted(directory.glob("*.sql")):
            source = migration.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if migration.name in applied:
                if applied[migration.name] != digest:
                    raise ValueError(f"Applied migration has changed: {migration.name}")
                continue
            sql = _migration_sql(source, migration.name)
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO baloot_migrations.applied (name, sha256) VALUES (%s,%s)",
                    (migration.name, digest),
                )
            print(f"applied {migration.name}")


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
