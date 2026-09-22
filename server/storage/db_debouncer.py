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

import threading
import time
from typing import Any, Callable, Dict, Optional


class DebouncedStatusUpdater:
    def __init__(
        self,
        update_fn: Callable[..., Any],
        min_interval_sec: float = 1.2,
        min_progress_delta: int = 5,
    ):
        """
        Args:
            update_fn: Callable with signature
                       (session_id, status, progress, stage, error, **extra)
            min_interval_sec: Minimum seconds between routine progress flushes.
            min_progress_delta: Minimum progress % jump to bypass time throttle.
        """
        self.update_fn = update_fn
        self.min_interval_sec = min_interval_sec
        self.min_progress_delta = min_progress_delta

        self._lock = threading.RLock()
        self._last_flush_time = 0.0
        self._last_flushed_status: Optional[str] = None
        self._last_flushed_stage: Optional[str] = None
        self._last_flushed_progress: Optional[int] = None

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
                self._execute_flush()
            else:
                self.throttled_calls += 1

    def flush(self) -> None:
        """Forces an immediate transmission of any pending buffered updates."""
        with self._lock:
            if self._has_pending:
                self._execute_flush()

    def _execute_flush(self) -> None:
        """Internal flush execution (must be called with _lock acquired)."""
        if not self._has_pending or not self._pending_session_id or not self._pending_status:
            return

        self.update_fn(
            self._pending_session_id,
            self._pending_status,
            progress=self._pending_progress,
            stage=self._pending_stage,
            error=self._pending_error,
            **self._pending_extra,
        )

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
