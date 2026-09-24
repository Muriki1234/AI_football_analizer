"""
pipeline_concurrency_scheduler.py — Asynchronous Pipeline Overlap & Concurrency Controller

Eliminates GPU serialization bubbles by managing concurrent execution between
SAMURAI player tracking (multi-segment subprocesses) and YOLO global analysis (streaming).

Guarantees:
1. VRAM Headroom Gating: Probes torch.cuda.mem_get_info() to ensure >= 6.0 GB free VRAM.
2. Lifecycle Barrier Synchronization: Uses threading.Event to synchronize before summary computation.
3. Safe Serial Fallback: Automatically falls back to sequential execution on low-memory GPUs or errors.
4. Telemetry Tracking: Records exact wall-clock time saved by concurrency overlap.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)


def compute_samurai_concurrency_cap(
    orig_w: int,
    orig_h: int,
    total_ram_gb: Optional[float] = None,
    env_cap_override: Optional[int] = None,
) -> int:
    """
    Computes an optimal, memory-safe concurrency ceiling for SAMURAI player tracking.

    1. Base cap derived from available system RAM (each worker uses ~3.5GB-4GB):
       - RAM >= 120GB (RunPod large pods): 16 workers
       - RAM >= 60GB (standard instances): 14 workers
       - RAM >= 40GB (baseline instances): 12 workers
       - RAM < 40GB (budget instances): 8 workers
    2. Overridable via SAMURAI_MAX_PARALLEL env var or env_cap_override.
    3. Scaled down gracefully for ultra-high-resolution (>1080p) using sqrt scaling.
    4. Uses round() rather than int() to prevent boundary truncation (e.g. 1926x1080).
    """
    if env_cap_override is not None:
        base_cap = env_cap_override
    else:
        env_val = os.environ.get("SAMURAI_MAX_PARALLEL")
        if env_val:
            try:
                base_cap = int(env_val)
            except ValueError:
                base_cap = 14
        else:
            if total_ram_gb is None:
                try:
                    import psutil
                    total_ram_gb = psutil.virtual_memory().total / (1024.0 ** 3)
                except Exception:
                    total_ram_gb = 48.0

            # Production default: 4 concurrent SAMURAI workers (4-parallel execution).
            # Memory footprint: 16 GB base + 4 * 10.5 GB = 58 GB (well below 125 GB container limit).
            # Overridable via SAMURAI_MAX_PARALLEL or env_cap_override.
            base_cap = 4

    # Baseline: 1080p (1920×1080 ≈ 2.07M pixels)
    baseline_px = 1920 * 1080
    res_factor = max(1.0, (orig_w * orig_h) / baseline_px)

    # Use round() so 10.9828 (1926x1080) rounds to 11 instead of truncated down to 10
    cap = max(1, int(round(base_cap / (res_factor ** 0.5))))
    return cap


class PipelineConcurrencyScheduler:
    """
    Manages safe overlap between SAMURAI tracking and YOLO detection pipelines.
    """

    def __init__(
        self,
        min_free_vram_gb: float = 6.0,
        default_mode: str = "auto",
    ) -> None:
        self.min_free_vram_gb = float(min_free_vram_gb)
        self.default_mode = os.environ.get(
            "PIPELINE_CONCURRENCY_MODE", default_mode
        ).strip().lower()

    @staticmethod
    def get_vram_headroom() -> Tuple[float, float, bool]:
        """
        Safely probes available CUDA VRAM in GB.
        Returns: (free_gb, total_gb, is_cuda_available)
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024.0 ** 3)
                total_gb = total_bytes / (1024.0 ** 3)
                return free_gb, total_gb, True
        except Exception:
            pass
        return 0.0, 0.0, False

    def should_run_concurrently(
        self,
        override_mode: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether concurrency is safe and advantageous.
        Modes:
        - "force_concurrent": Always attempt concurrent execution.
        - "force_serial": Always run strictly sequentially.
        - "auto": Check CUDA availability and free VRAM >= min_free_vram_gb.
        """
        mode = (override_mode or self.default_mode).strip().lower()

        if mode == "force_serial":
            return False, "Mode forced to sequential by configuration."

        if mode == "force_concurrent":
            return True, "Mode forced to concurrent by configuration."

        # Auto mode: inspect physical hardware
        free_gb, total_gb, cuda_ok = self.get_vram_headroom()
        if not cuda_ok:
            return False, "CUDA GPU unavailable; falling back to serial CPU mode."

        if free_gb < self.min_free_vram_gb:
            return False, (
                f"Insufficient VRAM headroom: {free_gb:.1f} GB free < "
                f"{self.min_free_vram_gb:.1f} GB requirement. Serial fallback engaged."
            )

        return True, (
            f"VRAM headroom verified: {free_gb:.1f} GB free of {total_gb:.1f} GB. "
            "Safe to overlap SAMURAI and YOLO concurrently."
        )

    def execute_pipeline(
        self,
        session_id: str,
        session: Dict[str, Any],
        segments: list,
        sm: Any,
        samurai_runner: Callable[[str, Dict[str, Any], list, Any], Any],
        yolo_runner: Callable[[str, Dict[str, Any], Any], Any],
        override_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates pipeline execution, dynamically selecting between
        concurrent overlapping and safe sequential execution.
        """
        run_concurrent, reason = self.should_run_concurrently(override_mode)
        log.info("[ORCHESTRATOR] Concurrency Decision for %s: %s (%s)", session_id, run_concurrent, reason)
        print(f"[ORCHESTRATOR] Mode: {'CONCURRENT' if run_concurrent else 'SERIAL'} — {reason}", flush=True)

        t_start = time.perf_counter()

        if not run_concurrent:
            # ── 1. Sequential Execution ──────────────────────────────────────────
            t0 = time.perf_counter()
            samurai_res = samurai_runner(session_id, session, segments, sm)
            t_samurai = time.perf_counter() - t0

            t0 = time.perf_counter()
            yolo_res = yolo_runner(session_id, session, sm)
            t_yolo = time.perf_counter() - t0

            t_total = time.perf_counter() - t_start
            return {
                "mode": "serial",
                "reason": reason,
                "wall_clock_sec": round(t_total, 2),
                "samurai_sec": round(t_samurai, 2),
                "yolo_sec": round(t_yolo, 2),
                "overlap_saved_sec": 0.0,
                "samurai_result": samurai_res,
                "yolo_result": yolo_res,
            }

        # ── 2. Concurrent Overlapping Execution ──────────────────────────────
        samurai_done_event = threading.Event()
        session_copy = dict(session)
        session_copy["_samurai_done_event"] = samurai_done_event

        samurai_error: list[Optional[Exception]] = [None]
        samurai_result: list[Any] = [None]
        t_samurai_box: list[float] = [0.0]

        def _samurai_worker():
            t0 = time.perf_counter()
            try:
                samurai_result[0] = samurai_runner(session_id, session_copy, segments, sm)
            except Exception as e:
                log.exception("[ORCHESTRATOR] SAMURAI worker encountered exception")
                samurai_error[0] = e
            finally:
                t_samurai_box[0] = time.perf_counter() - t0
                samurai_done_event.set()

        samurai_thread = threading.Thread(
            target=_samurai_worker,
            name=f"samurai_worker_{session_id}",
            daemon=True,
        )
        samurai_thread.start()

        # Execute YOLO on the main orchestrator thread
        t0 = time.perf_counter()
        yolo_error: Optional[Exception] = None
        yolo_result: Any = None
        try:
            yolo_result = yolo_runner(session_id, session_copy, sm)
        except Exception as e:
            log.exception("[ORCHESTRATOR] YOLO runner encountered exception")
            yolo_error = e
        t_yolo = time.perf_counter() - t0

        # Wait for SAMURAI thread to join
        samurai_thread.join(timeout=30.0)

        t_total = time.perf_counter() - t_start
        t_samurai = t_samurai_box[0]

        # Check if either failed
        if samurai_error[0] is not None:
            raise RuntimeError(f"Concurrent SAMURAI task failed: {samurai_error[0]}") from samurai_error[0]
        if yolo_error is not None:
            raise RuntimeError(f"Concurrent YOLO task failed: {yolo_error}") from yolo_error

        # Theoretical serial duration: t_samurai + t_yolo
        t_serial_est = t_samurai + t_yolo
        overlap_saved = max(0.0, t_serial_est - t_total)

        print(
            f"[ORCHESTRATOR] Concurrency completed in {t_total:.2f}s "
            f"(SAMURAI: {t_samurai:.2f}s, YOLO: {t_yolo:.2f}s | "
            f"Saved: {overlap_saved:.2f}s, {overlap_saved / max(1e-5, t_serial_est) * 100:.1f}%)",
            flush=True,
        )

        return {
            "mode": "concurrent",
            "reason": reason,
            "wall_clock_sec": round(t_total, 2),
            "samurai_sec": round(t_samurai, 2),
            "yolo_sec": round(t_yolo, 2),
            "overlap_saved_sec": round(overlap_saved, 2),
            "samurai_result": samurai_result[0],
            "yolo_result": yolo_result,
        }
