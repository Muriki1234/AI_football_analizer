"""
In-process pub/sub so pipeline threads can push status updates and SSE routes
can subscribe. Survives as long as the process does — no cross-worker
coordination because FastAPI runs single-worker per uvicorn process in this
deployment.

Every change to a session or task publishes an event. Subscribers get a queue
of events filtered by session_id.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class Event:
    session_id: str
    kind: str           # "session" | "task" | "heartbeat"
    data: dict[str, Any]
    ts: float

    def to_sse(self) -> str:
        return json.dumps({"kind": self.kind, "data": self.data, "ts": self.ts})


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called from FastAPI lifespan so threads can schedule on the main loop."""
        self._loop = loop

    def publish(self, session_id: str, kind: str, data: dict[str, Any]) -> None:
        """Thread-safe: pipeline workers call this from arbitrary threads."""
        evt = Event(session_id=session_id, kind=kind, data=data, ts=time.time())
        with self._lock:
            queues = list(self._subs.get(session_id, ()))
        if not queues or self._loop is None:
            return
        for q in queues:
            # Drop events if a slow subscriber has backed up beyond 128 items
            # — SSE connections are cheap to reopen and we'd rather lose heartbeats
            # than OOM.
            self._loop.call_soon_threadsafe(self._offer, q, evt)

    @staticmethod
    def _offer(queue: asyncio.Queue, evt: Event) -> None:
        try:
            queue.put_nowait(evt)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(evt)
            except Exception:
                pass

    async def subscribe(
        self, session_id: str, heartbeat_seconds: float = 15.0
    ) -> AsyncIterator[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=128)
        with self._lock:
            self._subs[session_id].add(q)
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=heartbeat_seconds)
                    yield evt
                except asyncio.TimeoutError:
                    yield Event(
                        session_id=session_id,
                        kind="heartbeat",
                        data={},
                        ts=time.time(),
                    )
        finally:
            with self._lock:
                self._subs[session_id].discard(q)
                if not self._subs[session_id]:
                    self._subs.pop(session_id, None)


bus = EventBus()
