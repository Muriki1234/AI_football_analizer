"""
/health  — liveness (server process is up)
/ready   — readiness (weights loaded, DB reachable)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from ..config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict:
    weights_ready = (
        os.environ.get("YOLO_MODEL_PATH") is not None
        and Path(os.environ.get("YOLO_MODEL_PATH", "")).exists()
    )
    db_ready = settings.db_path.parent.exists()
    return {
        "status": "ready" if weights_ready and db_ready else "degraded",
        "weights": weights_ready,
        "db": db_ready,
        "output_root": str(settings.output_root),
        "gpu_workers": settings.GPU_WORKERS,
        "io_workers": settings.IO_WORKERS,
    }
