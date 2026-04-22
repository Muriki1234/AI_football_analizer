"""
Bounded thread pools.

- GPU pool (1 thread): runs the YOLO + SAMURAI + rendering pipeline jobs.
  Anything that touches CUDA goes here. One slot = no VRAM contention.
- IO pool (N threads): downloads, trims, transcoding (ffmpeg subprocess).

The pipeline/tasks.py module is already thread-safe — it takes `sm` (session
manager) and writes back via update_status. We just schedule its entry
points here and let it log / publish its own progress.

Jobs carry a `task_id` and `session_id` so failures can be recorded against
the right row before the pool thread dies.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger(__name__)


class WorkerPool:
    def __init__(self, gpu_workers: int = 1, io_workers: int = 4) -> None:
        self._gpu = ThreadPoolExecutor(max_workers=gpu_workers, thread_name_prefix="gpu")
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="io")
        self._shutting_down = threading.Event()

    def submit_gpu(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_error: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ):
        return self._submit(self._gpu, fn, args, kwargs, on_error)

    def submit_io(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_error: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ):
        return self._submit(self._io, fn, args, kwargs, on_error)

    def _submit(
        self,
        pool: ThreadPoolExecutor,
        fn: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        on_error: Callable[[BaseException], None] | None,
    ):
        if self._shutting_down.is_set():
            raise RuntimeError("WorkerPool is shutting down; refusing new submissions.")

        def _run() -> Any:
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                log.exception("Pool job failed: %s", fn.__name__)
                if on_error:
                    try:
                        on_error(exc)
                    except Exception:
                        log.error("on_error callback itself raised:\n%s", traceback.format_exc())
                raise

        return pool.submit(_run)

    def shutdown(self, wait: bool = True, timeout: float | None = 10.0) -> None:
        self._shutting_down.set()
        self._gpu.shutdown(wait=wait, cancel_futures=True)
        self._io.shutdown(wait=wait, cancel_futures=True)


pool = WorkerPool()
