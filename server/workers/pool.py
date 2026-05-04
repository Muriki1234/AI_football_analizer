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
    def __init__(self, gpu_workers: int = 1, io_workers: int = 4,
                 quick_workers: int = 2) -> None:
        # _gpu  : long-running pipeline tasks (SAMURAI, full analysis) that
        #         hold the GPU for minutes. Single-threaded so we never have
        #         two heavy tasks fighting for VRAM.
        # _quick: fast GPU ops (detect-frame ~200ms). Lives on its own pool
        #         so it never queues behind a 10-minute analysis. CUDA will
        #         serialize kernel launches under contention but VRAM is
        #         plentiful on a 48 GB A40.
        # _io   : ffmpeg / disk / network. CPU-bound, no GPU.
        self._gpu = ThreadPoolExecutor(max_workers=gpu_workers, thread_name_prefix="gpu")
        self._quick = ThreadPoolExecutor(max_workers=quick_workers, thread_name_prefix="quick")
        self._io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="io")
        self._shutting_down = threading.Event()

    def submit_gpu(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_error: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ):
        return self._submit(self._gpu, fn, args, kwargs, on_error, max_queue_size=5)

    def submit_quick(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_error: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ):
        return self._submit(self._quick, fn, args, kwargs, on_error)

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
        max_queue_size: int | None = None,
    ):
        if self._shutting_down.is_set():
            raise RuntimeError("WorkerPool is shutting down; refusing new submissions.")
            
        if max_queue_size is not None and hasattr(pool, "_work_queue"):
            if pool._work_queue.qsize() >= max_queue_size:
                raise RuntimeError("Server is currently under heavy load (too many queued tasks). Please try again later.")

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
        self._quick.shutdown(wait=wait, cancel_futures=True)
        self._io.shutdown(wait=wait, cancel_futures=True)


pool = WorkerPool()
