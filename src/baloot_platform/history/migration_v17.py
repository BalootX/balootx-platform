from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from baloot_platform.version import __version__

from .canonical_store_v17 import CanonicalHistoryStore


@dataclass(frozen=True, slots=True)
class MigrationResult:
    game_id: str
    stream_id: str
    events: int
    snapshot_hash: str
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "gameId": self.game_id,
            "streamId": self.stream_id,
            "events": self.events,
            "snapshotHash": self.snapshot_hash,
            "valid": self.valid,
        }


def migrate_sessions_to_canonical(
    sessions: Iterable[Any],
    target: CanonicalHistoryStore,
    *,
    engine_version: str = __version__,
) -> tuple[MigrationResult, ...]:
    """Idempotently import typed legacy GameSession snapshots into V17 history."""
    results: list[MigrationResult] = []
    for session in sessions:
        stream_id = f"{session.id}:round:{session.state.round_number}"
        target.append(
            game_id=stream_id,
            events=tuple(session.state.events),
            profile_bundle_hash=str(session.bundle_hash),
            engine_version=engine_version,
        )
        snapshot = target.snapshot(
            game_id=stream_id,
            state=session.state.to_dict(reveal_hands=True),
            engine_version=engine_version,
            profile_bundle_hash=str(session.bundle_hash),
        )
        verification = target.verify(stream_id)
        results.append(MigrationResult(
            game_id=session.id,
            stream_id=stream_id,
            events=int(verification["event"].get("eventCount", 0)),
            snapshot_hash=snapshot.state_hash,
            valid=bool(verification["valid"]),
        ))
    return tuple(results)
