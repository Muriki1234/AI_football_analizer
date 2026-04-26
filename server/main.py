"""
FastAPI entry point. Wires the SQLite session manager, worker pool, HF weights
loader, event bus, and APScheduler cleanup job through the app lifespan.

Run locally:
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ensure_dirs, settings
from .events import bus
from .models.weights import WeightsError, ensure_weights
from .routes import analysis as analysis_routes
from .routes import health as health_routes
from .routes import sessions as session_routes
from .storage.db import SessionManager
from .workers.pool import WorkerPool

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s",
        stream=sys.stdout,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    log.info("── AI Football Assistant server starting ──")
    if not settings.API_KEY:
        log.warning(
            "API_KEY is empty — auth is DISABLED. "
            "Set API_KEY in server/.env before exposing this server."
        )

    ensure_dirs()
    log.info("workspace=%s", settings.WORKSPACE_DIR)
    log.info("output_root=%s", settings.output_root)

    # Pull weights from HF Hub (idempotent on warm Network Volumes).
    try:
        ensure_weights()
    except WeightsError as e:
        # Don't crash — /ready will report weights=false and routes will 503
        # once they try to load a model. This lets the server boot and expose
        # /health for diagnostics even if HF is having a bad day.
        log.error("Weights download failed: %s", e)

    # Singletons on app.state.
    sm = SessionManager(output_root=settings.output_root, db_path=settings.db_path)
    pool = WorkerPool(gpu_workers=settings.GPU_WORKERS, io_workers=settings.IO_WORKERS)
    app.state.session_manager = sm
    app.state.worker_pool = pool

    # Event bus needs the running loop to schedule callbacks from pipeline threads.
    bus.bind_loop(asyncio.get_running_loop())

    # Pre-warm YOLO on the GPU pool so CUDA JIT compiles at startup, not on
    # the first user request (which would cause a 1-2 min hang). Warm at the
    # same imgsz=1280 the streaming pipeline uses so the JIT cache actually
    # hits during analysis (not just during /detect-frame which uses 640).
    def _prewarm_yolo():
        import numpy as np
        from .pipeline.tasks import _get_yolo_model
        try:
            log.info("Pre-warming YOLO (first CUDA JIT may take ~60s)…")
            model = _get_yolo_model()
            dummy_lo = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_hi = np.zeros((1280, 1280, 3), dtype=np.uint8)
            # Warm both code paths (detect-frame uses 640, pipeline uses 1280).
            model.predict([dummy_lo], verbose=False)
            model.predict([dummy_hi], verbose=False, imgsz=1280, half=True)
            log.info("YOLO pre-warm done.")
        except Exception as exc:
            # Non-fatal: detect-frame / analyze will still try and surface a
            # clear FileNotFoundError if weights are missing.
            log.warning("YOLO pre-warm failed (non-fatal): %s", exc)

    pool.submit_gpu(_prewarm_yolo)

    # Scheduled session cleanup.
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        _cleanup_expired_sessions,
        trigger="interval",
        minutes=settings.CLEANUP_INTERVAL_MINUTES,
        args=[sm],
        id="session_cleanup",
        max_instances=1,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    log.info(
        "scheduler: session cleanup every %d min, TTL=%dh",
        settings.CLEANUP_INTERVAL_MINUTES, settings.SESSION_TTL_HOURS,
    )

    log.info("── ready on %s:%d ──", settings.HOST, settings.PORT)
    try:
        yield
    finally:
        log.info("── shutting down ──")
        scheduler.shutdown(wait=False)
        pool.shutdown(wait=True, timeout=10.0)
        sm.close()


def _cleanup_expired_sessions(sm: SessionManager) -> None:
    import shutil

    expired = sm.expired_sessions(settings.SESSION_TTL_HOURS)
    if not expired:
        return
    log.info("cleanup: %d expired session(s) to purge", len(expired))
    for sid in expired:
        try:
            shutil.rmtree(sm.session_output_dir(sid), ignore_errors=True)
            up = settings.upload_root / sid
            if up.exists():
                shutil.rmtree(up, ignore_errors=True)
            sm.delete_session(sid)
        except Exception as exc:
            log.warning("cleanup: failed to purge %s: %s", sid, exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Football Assistant",
        version="2.0.0",
        description="RunPod-native football video analysis API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.include_router(health_routes.router)
    app.include_router(session_routes.router)
    app.include_router(session_routes.events_router)
    app.include_router(analysis_routes.router)
    app.include_router(analysis_routes.files_router)
    return app


app = create_app()
