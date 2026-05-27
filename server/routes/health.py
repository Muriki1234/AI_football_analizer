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
    # 用真正的 resolver 检查 weights，而不是只看 env 变量 ——
    # pipeline 里走的是 get_yolo_model_path() 的 fallback 链，
    # 之前 /ready 只看 YOLO_MODEL_PATH env 容易给出假阴性/假阳性。
    weights_ready = False
    try:
        from ..pipeline.tasks import get_yolo_model_path, get_keypoint_model_path
        ypath = get_yolo_model_path()
        kpath = get_keypoint_model_path()
        weights_ready = bool(ypath and Path(ypath).exists()
                             and kpath and Path(kpath).exists())
    except Exception:
        weights_ready = False
    db_ready = settings.db_path.parent.exists()
    return {
        "status": "ready" if weights_ready and db_ready else "degraded",
        "weights": weights_ready,
        "db": db_ready,
        "output_root": str(settings.output_root),
        "gpu_workers": settings.GPU_WORKERS,
        "io_workers": settings.IO_WORKERS,
    }
