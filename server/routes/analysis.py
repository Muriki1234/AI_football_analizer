"""
Analysis pipeline routes: kick off global analysis, queue features, fetch
task state, serve artifacts.

Feature dispatch table maps the short name the frontend uses (e.g. "heatmap")
to the right pipeline task entry point.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..auth import require_api_key, require_api_key_or_query
from ..config import settings
from ..deps import get_session_manager, get_worker_pool
from ..pipeline import tasks as pipeline_tasks
from ..storage.db import SessionManager
from ..workers.pool import WorkerPool

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["analysis"], dependencies=[Depends(require_api_key)])

# Separate router for endpoints that need ?key= query auth (img/video tags can't
# send headers). Mounted at the same prefix; the global header-based auth on the
# main router does NOT apply here.
files_router = APIRouter(prefix="/api/sessions", tags=["analysis-files"])


# ── Feature table ────────────────────────────────────────────────────────────

FEATURE_TASKS: dict[str, Callable[..., Any]] = {
    "heatmap":         pipeline_tasks.run_heatmap,
    "speed_chart":     pipeline_tasks.run_speed_chart,
    "possession":      pipeline_tasks.run_possession_stats,
    "minimap_replay":  pipeline_tasks.run_minimap_replay,
    "full_replay":     pipeline_tasks.run_full_replay,
    "sprint_analysis": pipeline_tasks.run_sprint_analysis,
    "defensive_line":  pipeline_tasks.run_defensive_line,
    "ai_summary":      pipeline_tasks.run_ai_summary,
}


class TrackPayload(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    frame: int = Field(ge=0, default=0)


class DetectFramePayload(BaseModel):
    frame: int = Field(ge=0, default=0)


class QueuedResponse(BaseModel):
    task_id: str
    status: str = "queued"


# ── First-frame player detection (sync, fast) ────────────────────────────────


@router.post("/{session_id}/detect-frame")
async def detect_frame(
    session_id: str,
    payload: DetectFramePayload | None = None,
    sm: SessionManager = Depends(get_session_manager),
    pool: WorkerPool = Depends(get_worker_pool),
) -> dict:
    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    if not s.get("video_path") or not Path(s["video_path"]).exists():
        raise HTTPException(status_code=400, detail="Session has no video")

    frame_idx = payload.frame if payload else 0

    import asyncio
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        pool._gpu,  # YOLO uses GPU; reuse the GPU pool to avoid concurrent CUDA ctx
        pipeline_tasks.detect_frame_players,
        session_id, s, frame_idx, sm,
    )
    try:
        result = await fut
    except Exception as e:
        log.exception("detect_frame failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    # Build a frame URL the frontend can render
    rel = result.get("annotated_frame_path", "first_frame.jpg")
    result["annotated_frame_url"] = f"/api/sessions/{session_id}/files/{rel}"
    return result


# ── Start global analysis ────────────────────────────────────────────────────


@router.post("/{session_id}/analyze", response_model=QueuedResponse)
async def start_analysis(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
    pool: WorkerPool = Depends(get_worker_pool),
) -> QueuedResponse:
    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    if not s.get("video_path") or not Path(s["video_path"]).exists():
        raise HTTPException(status_code=400, detail="Session has no video")

    sm.update_status(session_id, "analyzing", progress=0, stage="queued")

    def _on_error(exc: BaseException) -> None:
        sm.update_status(session_id, "analysis_failed", error=str(exc))

    pool.submit_gpu(pipeline_tasks.run_global_analysis, session_id, s, sm, on_error=_on_error)
    return QueuedResponse(task_id=session_id, status="analyzing")


# ── SAMURAI tracking ─────────────────────────────────────────────────────────


@router.post("/{session_id}/track", response_model=QueuedResponse)
async def start_tracking(
    session_id: str,
    payload: TrackPayload,
    sm: SessionManager = Depends(get_session_manager),
    pool: WorkerPool = Depends(get_worker_pool),
) -> QueuedResponse:
    if not settings.SAMURAI_SCRIPT:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SAMURAI tracking is not configured on this server.",
        )

    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")

    x1, y1, x2, y2 = payload.bbox
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="Invalid bbox")

    s_merged = {
        **s,
        "selected_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "start_frame": payload.frame,
    }
    sm.update_status(session_id, "tracking", progress=0, stage="samurai_queued",
                     selected_bbox=s_merged["selected_bbox"],
                     start_frame=payload.frame)

    def _on_error(exc: BaseException) -> None:
        # Whichever phase raised, mark the appropriate failure status.
        cur = sm.get_session(session_id) or {}
        st = cur.get("status") or ""
        if st.startswith("track"):
            sm.update_status(session_id, "tracking_failed", error=str(exc))
        else:
            sm.update_status(session_id, "analysis_failed", error=str(exc))

    def _track_then_analyze() -> None:
        # Phase 1: SAMURAI tracking (catches its own errors → tracking_failed).
        pipeline_tasks.run_samurai_tracking(session_id, s_merged, sm)
        s_after = sm.get_session(session_id) or {}
        if s_after.get("status") != "tracking_done":
            log.info("[chain] SAMURAI did not complete (status=%s); skipping analysis.",
                     s_after.get("status"))
            return
        # Phase 2: global analysis (also catches its own errors → analysis_failed).
        pipeline_tasks.run_global_analysis(session_id, s_after, sm)

    pool.submit_gpu(_track_then_analyze, on_error=_on_error)
    return QueuedResponse(task_id=session_id, status="tracking")


# ── Feature generation ───────────────────────────────────────────────────────


@router.post("/{session_id}/features/{feature}", response_model=QueuedResponse)
async def queue_feature(
    session_id: str,
    feature: str,
    sm: SessionManager = Depends(get_session_manager),
    pool: WorkerPool = Depends(get_worker_pool),
) -> QueuedResponse:
    if feature not in FEATURE_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature '{feature}'. Available: {sorted(FEATURE_TASKS)}",
        )

    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    if s.get("status") not in ("analysis_done", "analyzing"):
        # Allow queuing while analyzing so the UI can pre-queue features;
        # tasks.py will refuse to run until data is ready and update the task row.
        pass

    fn = FEATURE_TASKS[feature]
    task_id = sm.create_task(session_id, feature)

    def _on_error(exc: BaseException) -> None:
        sm.update_task(session_id, task_id, status="failed", error=str(exc))

    # AI summary hits an external API, not the GPU — send it to the IO pool.
    target_pool = pool.submit_io if feature == "ai_summary" else pool.submit_gpu
    target_pool(fn, session_id, s, task_id, sm, on_error=_on_error)
    return QueuedResponse(task_id=task_id, status="queued")


# ── Task + summary readback (SSE is preferred; these remain for recovery) ───


@router.get("/{session_id}/tasks")
async def list_tasks(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
) -> list[dict]:
    if not sm.get_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")
    return sm.list_tasks(session_id)


@router.get("/{session_id}/tasks/{task_id}")
async def get_task(
    session_id: str,
    task_id: str,
    sm: SessionManager = Depends(get_session_manager),
) -> dict:
    t = sm.get_task(session_id, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Unknown task")
    return t


@router.get("/{session_id}/summary")
async def get_summary(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
) -> dict:
    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    return {
        "session": s,
        "tasks": sm.list_tasks(session_id),
    }


# ── Artifact serving ─────────────────────────────────────────────────────────


@files_router.get("/{session_id}/files/{path:path}", dependencies=[Depends(require_api_key_or_query)])
async def serve_artifact(
    session_id: str,
    path: str,
    sm: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    if not sm.get_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")

    base = sm.session_output_dir(session_id).resolve()
    target = (base / path).resolve()
    # Prevent escaping the session directory via `../..`
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    mime, _ = mimetypes.guess_type(target.name)
    return FileResponse(str(target), media_type=mime or "application/octet-stream")
