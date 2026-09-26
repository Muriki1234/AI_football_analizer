"""
db_debouncer.py — Telemetry & Progress Debouncer for Cloud Database Operations

Prevents synchronous HTTP network request storms (e.g. Supabase SELECT+UPDATE round-trips)
during high-cadence ML video processing loops.

Rules:
1. Immediate Flush: Status changes, stage transitions, error reports, and explicit flush/close.
2. Throttled Flush: Routine progress ticks (e.g. 1% increments) are throttled to at most
   one update per min_interval_sec (default: 1.2s) unless progress delta >= min_progress_delta (default: 5%).
3. Extra Payload Consolidation: Intermediate 'extra' key-value updates are coalesced in memory
   so only one consolidated payload is transmitted to the database.
4. Thread-Safe: Protected by an internal reentrant lock.
"""

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional


class DebouncedStatusUpdater:
    def __init__(
        self,
        update_fn: Callable[..., Any],
        min_interval_sec: float = 1.2,
        min_progress_delta: int = 5,
        async_dispatch: bool = False,
    ):
        """
        Args:
            update_fn: Callable with signature
                       (session_id, status, progress, stage, error, **extra)
            min_interval_sec: Minimum seconds between routine progress flushes.
            min_progress_delta: Minimum progress % jump to bypass time throttle.
            async_dispatch: When True, dispatches routine updates to a background
                            worker thread so caller (e.g. YOLO loop) does not block on HTTP.
        """
        self.update_fn = update_fn
        self.min_interval_sec = min_interval_sec
        self.min_progress_delta = min_progress_delta
        self.async_dispatch = async_dispatch

        self._lock = threading.RLock()
        self._last_flush_time = 0.0
        self._last_flushed_status: Optional[str] = None
        self._last_flushed_stage: Optional[str] = None
        self._last_flushed_progress: Optional[int] = None

        if self.async_dispatch:
            self._queue: queue.Queue = queue.Queue(maxsize=20)
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="db_debouncer_async_worker",
                daemon=True,
            )
            self._worker_thread.start()
        else:
            self._queue = None
            self._worker_thread = None

        # Pending dirty state
        self._pending_session_id: Optional[str] = None
        self._pending_status: Optional[str] = None
        self._pending_progress: Optional[int] = None
        self._pending_stage: Optional[str] = None
        self._pending_error: Optional[str] = None
        self._pending_extra: Dict[str, Any] = {}
        self._has_pending = False

        # Telemetry metrics
        self.total_calls = 0
        self.flushed_calls = 0
        self.throttled_calls = 0

    @staticmethod
    def _normalize_stage(stage: Optional[str]) -> str:
        if not stage:
            return ""
        # Strip dynamic counts/ETA in parentheses: e.g. "streaming_analysis (150/25316 frames)" -> "streaming_analysis"
        return stage.split("(")[0].strip()

    def update(
        self,
        session_id: str,
        status: str,
        progress: Optional[int] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Record a status/progress update. Evaluates whether to flush immediately
        or buffer until the debounce window expires.
        """
        with self._lock:
            self.total_calls += 1
            now = time.time()

            # Merge into pending state
            self._pending_session_id = session_id
            self._pending_status = status
            if progress is not None:
                self._pending_progress = progress
            if stage is not None:
                self._pending_stage = stage
            if error is not None:
                self._pending_error = error
            self._pending_extra.update(extra)
            self._has_pending = True

            # Check critical immediate bypass conditions
            is_critical = False
            if error is not None:
                is_critical = True
            elif status != self._last_flushed_status:
                is_critical = True
            elif stage is not None and self._normalize_stage(stage) != self._normalize_stage(self._last_flushed_stage):
                is_critical = True
            elif status in ("done", "failed", "analysis_failed", "tracking_failed"):
                is_critical = True

            # Check progress delta condition
            progress_jumped = False
            if (
                self._pending_progress is not None
                and self._last_flushed_progress is not None
                and abs(self._pending_progress - self._last_flushed_progress) >= self.min_progress_delta
            ):
                progress_jumped = True

            time_elapsed = (now - self._last_flush_time) >= self.min_interval_sec

            if is_critical or progress_jumped or time_elapsed:
                self._execute_flush(is_critical=is_critical)
            else:
                self.throttled_calls += 1

    def _worker_loop(self) -> None:
        """Background daemon thread worker processing non-critical status flushes."""
        while True:
            try:
                task = self._queue.get()
                if task is None:
                    self._queue.task_done()
                    break
                session_id, status, progress, stage, error, extra = task
                try:
                    self.update_fn(
                        session_id,
                        status,
                        progress=progress,
                        stage=stage,
                        error=error,
                        **extra,
                    )
                except Exception:
                    pass
                finally:
                    self._queue.task_done()
            except Exception:
                pass

    def flush(self) -> None:
        """Forces an immediate transmission of any pending buffered updates and drains queue."""
        with self._lock:
            if self._has_pending:
                self._execute_flush(is_critical=True)

        if self.async_dispatch and self._queue is not None:
            self._queue.join()

    def _execute_flush(self, is_critical: bool = False) -> None:
        """Internal flush execution (must be called with _lock acquired)."""
        if not self._has_pending or not self._pending_session_id or not self._pending_status:
            return

        sid = self._pending_session_id
        stat = self._pending_status
        prog = self._pending_progress
        stg = self._pending_stage
        err = self._pending_error
        ext = dict(self._pending_extra)

        if is_critical or not self.async_dispatch or self._queue is None:
            # Synchronous direct execution for critical state transitions
            self.update_fn(
                sid,
                stat,
                progress=prog,
                stage=stg,
                error=err,
                **ext,
            )
        else:
            # Asynchronous background execution: zero blocking on caller thread
            task = (sid, stat, prog, stg, err, ext)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except Exception:
                    pass
            try:
                self._queue.put_nowait(task)
            except Exception:
                # Fallback to direct synchronous execution if queue full/error
                self.update_fn(sid, stat, progress=prog, stage=stg, error=err, **ext)

        self._last_flush_time = time.time()
        self._last_flushed_status = self._pending_status
        self._last_flushed_stage = self._pending_stage
        self._last_flushed_progress = self._pending_progress

        # Reset dirty state
        self._pending_extra = {}
        self._pending_error = None
        self._has_pending = False
        self.flushed_calls += 1

    def __enter__(self) -> "DebouncedStatusUpdater":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.flush()
