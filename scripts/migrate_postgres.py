from __future__ import annotations

import hashlib
import os
from pathlib import Path


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
            sql = source.decode("utf-8").strip()
            # Preserve the old files and commit each migration with its receipt.
            if sql.startswith("BEGIN;") and sql.endswith("COMMIT;"):
                sql = sql.removeprefix("BEGIN;").removesuffix("COMMIT;")
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
