from __future__ import annotations

import asyncio
import threading
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class LiveEventBroker(Protocol):
    """Disposable notification projection for live room events.

    Durable truth remains in LiveStore. A broker only wakes connected clients;
    reconnecting clients always recover events by sequence from the durable room.
    """

    def publish(self, room_id: str, sequence: int) -> None: ...

    async def wait(self, room_id: str, after_sequence: int, timeout: float = 5.0) -> int: ...

    def close(self) -> None: ...


class InMemoryLiveEventBroker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: dict[str, int] = {}
        self._closed = False

    def publish(self, room_id: str, sequence: int) -> None:
        with self._condition:
            self._latest[room_id] = max(int(sequence), self._latest.get(room_id, 0))
            self._condition.notify_all()

    def _wait_sync(self, room_id: str, after_sequence: int, timeout: float) -> int:
        deadline = time.monotonic() + max(0.01, float(timeout))
        with self._condition:
            while not self._closed and self._latest.get(room_id, 0) <= int(after_sequence):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._latest.get(room_id, int(after_sequence))

    async def wait(self, room_id: str, after_sequence: int, timeout: float = 5.0) -> int:
        return await asyncio.to_thread(self._wait_sync, room_id, after_sequence, timeout)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class RedisLiveEventBroker:
    """Optional multi-node wake-up broker; Redis is never authoritative."""

    def __init__(self, url: str, *, prefix: str = "baloot:live") -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError('Install with: pip install "baloot-platform[redis]"') from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix.rstrip(":")

    def _key(self, room_id: str) -> str:
        return f"{self.prefix}:{room_id}"

    def publish(self, room_id: str, sequence: int) -> None:
        key = self._key(room_id)
        pipeline = self._redis.pipeline()
        pipeline.set(key, int(sequence), ex=86_400)
        pipeline.publish(key, int(sequence))
        pipeline.execute()

    def _wait_sync(self, room_id: str, after_sequence: int, timeout: float) -> int:
        key = self._key(room_id)
        current = int(self._redis.get(key) or 0)
        if current > int(after_sequence):
            return current
        pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(key)
        deadline = time.monotonic() + max(0.01, float(timeout))
        try:
            while time.monotonic() < deadline:
                message = pubsub.get_message(timeout=min(0.25, deadline - time.monotonic()))
                if message and int(message.get("data") or 0) > int(after_sequence):
                    return int(message["data"])
            return int(self._redis.get(key) or after_sequence)
        finally:
            pubsub.close()

    async def wait(self, room_id: str, after_sequence: int, timeout: float = 5.0) -> int:
        return await asyncio.to_thread(self._wait_sync, room_id, after_sequence, timeout)

    def close(self) -> None:
        self._redis.close()
