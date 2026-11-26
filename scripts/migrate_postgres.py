from __future__ import annotations

import os
import re
from pathlib import Path


_CONCURRENT_INDEX = re.compile(
    r"CREATE INDEX CONCURRENTLY IF NOT EXISTS ([a-z][a-z0-9_]*)\s+ON ([a-z][a-z0-9_]*)\."
)


def _apply_migration(conn, migration: Path) -> None:
    sql = migration.read_text(encoding="utf-8")
    if not sql.startswith("CREATE INDEX CONCURRENTLY "):
        conn.execute(sql)
        return
    match = _CONCURRENT_INDEX.match(sql)
    if match is None or sql.count(";") != 1 or not sql.rstrip().endswith(";"):
        raise ValueError(f"{migration.name} must contain one schema-qualified concurrent index build")
    index, schema = match.groups()
    qualified = f"{schema}.{index}"
    conn.commit()
    conn.autocommit = True
    try:
        conn.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (qualified,))
        try:
            # An interrupted build can leave an invalid index that IF NOT EXISTS would skip.
            valid = conn.execute(
                "SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(%s)", (qualified,)
            ).fetchone()
            if valid == (False,):
                conn.execute(f"DROP INDEX CONCURRENTLY {qualified}")
            conn.execute(sql)
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (qualified,))
    finally:
        conn.autocommit = False


def main() -> None:
    dsn = os.environ.get("BALOOT_DATABASE_URL", "").strip()
    if not dsn:
        raise SystemExit("BALOOT_DATABASE_URL is required")
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("install baloot-platform[postgres]") from exc
    root = Path(os.environ.get("BALOOT_ROOT", Path(__file__).resolve().parents[1]))
    migrations = sorted((root / "migrations").glob("*.sql"))
    with psycopg.connect(dsn) as conn:
        for migration in migrations:
            _apply_migration(conn, migration)
            print(f"applied {migration.name}")
    print("Baloot PostgreSQL schema is current")


if __name__ == "__main__":
    main()
