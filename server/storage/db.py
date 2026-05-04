"""
SQLite-backed session + task store. Keeps the exact same method signatures as
the legacy in-memory SessionManager so pipeline/tasks.py works unchanged.

Every mutating call also publishes an event on the global EventBus so SSE
subscribers get live progress without polling.

Serialisation strategy: sessions and tasks both have a handful of hot columns
(status, progress, stage, error, timestamps) and a catch-all `extra_json`
column for everything else the pipeline writes via `update_status(**extra)`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from ..events import bus

SESSION_COLUMNS = {
    "session_id", "video_path", "status", "progress", "stage", "error",
    "created_at", "updated_at",
}
TASK_COLUMNS = {
    "task_id", "task_type", "status", "progress", "result", "file_path",
    "url", "error", "created_at", "finished_at",
}


class SessionManager:
    """Drop-in replacement for the legacy in-memory manager.

    Thread-safe: one connection per thread (check_same_thread=False with a
    single RLock around all writes keeps things simple and fast enough for
    the single-GPU workload).
    """

    def __init__(self, output_root: Path, db_path: Path | None = None):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else self.output_root.parent / "sessions.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    video_path  TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'uploaded',
                    progress    INTEGER NOT NULL DEFAULT 0,
                    stage       TEXT NOT NULL DEFAULT '',
                    error       TEXT,
                    extra_json  TEXT NOT NULL DEFAULT '{}',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    session_id   TEXT NOT NULL,
                    task_id      TEXT NOT NULL,
                    task_type    TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'queued',
                    progress     INTEGER NOT NULL DEFAULT 0,
                    result       TEXT,
                    file_path    TEXT,
                    url          TEXT,
                    error        TEXT,
                    created_at   TEXT NOT NULL,
                    finished_at  TEXT,
                    PRIMARY KEY (session_id, task_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_created
                    ON sessions(created_at);
            """)

    # ── Session CRUD ─────────────────────────────────────────────────────────

    def create_session(self, session_id: str, video_path: str) -> None:
        now = _utcnow_iso()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, video_path, status, progress, stage, error, extra_json, created_at, updated_at)
                   VALUES (?, ?, 'uploaded', 0, '', NULL, '{}', ?, ?)""",
                (session_id, video_path, now, now),
            )
        bus.publish(session_id, "session", self.get_session(session_id) or {})

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def update_status(
        self,
        session_id: str,
        status: str,
        progress: int | None = None,
        stage: str | None = None,
        error: str | None = None,
        **extra: Any,
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT extra_json FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if not row:
                return
            extra_current = json.loads(row["extra_json"] or "{}")
            extra_current.update(_jsonable(extra))
            sets: list[str] = ["status=?", "extra_json=?", "updated_at=?"]
            params: list[Any] = [status, json.dumps(extra_current), _utcnow_iso()]
            if progress is not None:
                sets.append("progress=?")
                params.append(progress)
            if stage is not None:
                sets.append("stage=?")
                params.append(stage)
            if error is not None:
                sets.append("error=?")
                params.append(error)
            params.append(session_id)
            self._conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE session_id=?", params
            )
        snap = self.get_session(session_id)
        if snap:
            bus.publish(session_id, "session", snap)

    # ── Task CRUD ─────────────────────────────────────────────────────────────

    def create_task(self, session_id: str, task_type: str) -> str:
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        now = _utcnow_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO tasks
                   (session_id, task_id, task_type, status, progress, created_at)
                   VALUES (?, ?, ?, 'queued', 0, ?)""",
                (session_id, task_id, task_type, now),
            )
        snap = self.get_task(session_id, task_id)
        if snap:
            bus.publish(session_id, "task", snap)
        return task_id

    def get_task(self, session_id: str, task_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE session_id=? AND task_id=?",
                (session_id, task_id),
            ).fetchone()
        return _row_to_task(row) if row else None

    def list_tasks(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE session_id=? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def clear_tasks(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tasks WHERE session_id=?", (session_id,))

    def update_task(self, session_id: str, task_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        allowed = {"status", "progress", "result", "file_path", "url", "error"}
        sets: list[str] = []
        params: list[Any] = []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            sets.append(f"{k}=?")
            params.append(json.dumps(v) if k == "result" and not isinstance(v, str) else v)
        if not sets:
            return
        if kwargs.get("status") in ("done", "failed"):
            sets.append("finished_at=?")
            params.append(_utcnow_iso())
        params.extend([session_id, task_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE session_id=? AND task_id=?",
                params,
            )
        snap = self.get_task(session_id, task_id)
        if snap:
            bus.publish(session_id, "task", snap)

    def session_output_dir(self, session_id: str) -> Path:
        p = self.output_root / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Lifecycle / cleanup ──────────────────────────────────────────────────

    def expired_sessions(self, ttl_hours: int) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id FROM sessions WHERE created_at < ?", (cutoff,)
            ).fetchall()
        return [r["session_id"] for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tasks WHERE session_id=?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

    def cleanup_zombies(self, timeout_minutes: int = 60) -> None:
        """Mark sessions and tasks stuck in active states for too long as failed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)).isoformat()
        now = _utcnow_iso()
        with self._lock:
            # Clean up zombie sessions
            self._conn.execute(
                """UPDATE sessions 
                   SET status='analysis_failed', error='Task timed out (zombie watcher)' 
                   WHERE status IN ('analyzing', 'tracking', 'queued') AND updated_at < ?""",
                (cutoff,)
            )
            # Clean up zombie tasks
            self._conn.execute(
                """UPDATE tasks 
                   SET status='failed', error='Task timed out (zombie watcher)', finished_at=? 
                   WHERE status IN ('running', 'queued') AND created_at < ?""",
                (now, cutoff)
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_session(row: sqlite3.Row) -> dict:
    base = {k: row[k] for k in row.keys() if k != "extra_json"}
    extra = json.loads(row["extra_json"] or "{}")
    base.update(extra)
    return base


def _row_to_task(row: sqlite3.Row) -> dict:
    d = {k: row[k] for k in row.keys()}
    if d.get("result"):
        try:
            d["result"] = json.loads(d["result"])
        except Exception:
            pass
    return d


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce Path objects and other non-JSON types so extra_json round-trips."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


__all__ = ["SessionManager"]
