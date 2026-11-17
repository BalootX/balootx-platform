#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baloot_platform.history.canonical_store_v17 import CanonicalHistoryStore
from baloot_platform.history.migration_v17 import migrate_sessions_to_canonical
from baloot_platform.storage.game_store import SQLiteGameStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate typed Baloot game snapshots into the canonical V17 event store")
    parser.add_argument("--source", required=True, help="Legacy/current SQLite game-store path")
    parser.add_argument("--target", required=True, help="Canonical SQLite store path")
    parser.add_argument("--output", required=True, help="Migration report JSON")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    source = SQLiteGameStore(args.source)
    target = CanonicalHistoryStore(args.target)
    sessions = []
    for item in source.list_games(limit=args.limit):
        session = source.load(item["gameId"])
        if session is not None:
            sessions.append(session)
    results = migrate_sessions_to_canonical(sessions, target)
    payload = {
        "schemaVersion": 1,
        "status": "passed" if all(item.valid for item in results) else "failed",
        "source": str(Path(args.source).resolve()),
        "target": str(Path(args.target).resolve()),
        "games": [item.to_dict() for item in results],
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
