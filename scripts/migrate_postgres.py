from __future__ import annotations
import os
from pathlib import Path

def main() -> None:
    dsn=os.environ.get('BALOOT_DATABASE_URL','').strip()
    if not dsn: raise SystemExit('BALOOT_DATABASE_URL is required')
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit('install baloot-platform[postgres]') from exc
    root=Path(os.environ.get('BALOOT_ROOT',Path(__file__).resolve().parents[1]))
    migrations=sorted((root/'migrations').glob('*.sql'))
    with psycopg.connect(dsn) as conn:
        for migration in migrations:
            conn.execute(migration.read_text(encoding='utf-8'))
            print(f'applied {migration.name}')
    print('Baloot PostgreSQL schema is current')

if __name__=='__main__': main()
