"""
Session lifecycle routes: create / chunked upload / trim / snapshot / SSE.

Upload strategy: the frontend can either POST multipart for small files or
stream chunks (5 MB default) via PUT /sessions/{id}/chunks/{index}. Completion
atomically renames the assembled file into place.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..auth import require_api_key
from ..config import settings
from ..deps import get_session_manager
from ..events import bus
from ..storage.db import SessionManager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)])


# ── Models ───────────────────────────────────────────────────────────────────


class CreateSessionResponse(BaseModel):
    session_id: str


class CompleteUploadPayload(BaseModel):
    total_chunks: int = Field(gt=0, le=10_000)
    filename: str = Field(min_length=1, max_length=255)


class TrimPayload(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _session_upload_dir(session_id: str) -> Path:
    p = settings.upload_root / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_filename(name: str) -> str:
    # Strip directory traversal attempts; keep extension.
    return Path(name).name


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    file: UploadFile | None = File(default=None),
    sm: SessionManager = Depends(get_session_manager),
) -> CreateSessionResponse:
    """Create a session. If `file` is provided, upload is one-shot; otherwise
    the client will use the chunked upload routes below."""
    session_id = uuid.uuid4().hex[:12]
    upload_dir = _session_upload_dir(session_id)

    if file is not None:
        filename = _safe_filename(file.filename or "video.mp4")
        dest = upload_dir / filename
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    out.write(chunk)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Disk write failed: {e}") from e
        sm.create_session(session_id, str(dest))
        sm.update_status(session_id, "uploaded", progress=100, stage="upload_complete")
    else:
        # Chunked upload will register the final path on /complete.
        sm.create_session(session_id, "")
        sm.update_status(session_id, "uploading", progress=0, stage="awaiting_chunks")

    return CreateSessionResponse(session_id=session_id)


@router.put("/{session_id}/chunks/{index}")
async def upload_chunk(
    session_id: str,
    index: int,
    request: Request,
    sm: SessionManager = Depends(get_session_manager),
) -> dict:
    if not sm.get_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")
    if index < 0 or index > 10_000:
        raise HTTPException(status_code=400, detail="Chunk index out of range")

    chunks_dir = _session_upload_dir(session_id) / ".chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunks_dir / f"{index:06d}.part"

    try:
        with chunk_path.open("wb") as out:
            async for data in request.stream():
                out.write(data)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Disk write failed: {e}") from e

    return {"ok": True, "index": index, "size": chunk_path.stat().st_size}


@router.post("/{session_id}/complete")
async def complete_upload(
    session_id: str,
    payload: CompleteUploadPayload,
    sm: SessionManager = Depends(get_session_manager),
) -> dict:
    if not sm.get_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")

    upload_dir = _session_upload_dir(session_id)
    chunks_dir = upload_dir / ".chunks"
    if not chunks_dir.is_dir():
        raise HTTPException(status_code=400, detail="No chunks uploaded")

    filename = _safe_filename(payload.filename)
    final_path = upload_dir / filename
    tmp_path = upload_dir / f".{filename}.tmp"

    # Verify every chunk is present before starting — fail fast.
    for i in range(payload.total_chunks):
        if not (chunks_dir / f"{i:06d}.part").exists():
            raise HTTPException(status_code=400, detail=f"Missing chunk {i}")

    try:
        with tmp_path.open("wb") as out:
            for i in range(payload.total_chunks):
                with (chunks_dir / f"{i:06d}.part").open("rb") as part:
                    shutil.copyfileobj(part, out, length=1024 * 1024)
        tmp_path.replace(final_path)
    except OSError as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Assembly failed: {e}") from e

    shutil.rmtree(chunks_dir, ignore_errors=True)

    sm.update_status(
        session_id,
        "uploaded",
        progress=100,
        stage="upload_complete",
        video_path=str(final_path),
    )
    return {"ok": True, "video_path": str(final_path), "size": final_path.stat().st_size}


@router.post("/{session_id}/trim")
async def trim(
    session_id: str,
    payload: TrimPayload,
    sm: SessionManager = Depends(get_session_manager),
) -> dict:
    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    src = _resolve_video_path(s)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source video missing on disk")
    if payload.end <= payload.start:
        raise HTTPException(status_code=400, detail="end must be greater than start")

    duration = payload.end - payload.start
    trimmed = src.with_name(f"{src.stem}.trimmed{src.suffix}")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{payload.start}",
        "-i", str(src),
        "-t", f"{duration}",
        "-c", "copy",
        str(trimmed),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not installed on server")
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg trim failed: %s", e.stderr.decode(errors="replace"))
        raise HTTPException(status_code=500, detail="Trim failed; see server logs")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Trim timed out (>120s)")

    sm.update_status(
        session_id, s["status"], stage="trimmed",
        video_path=str(trimmed),
        trim_start=payload.start,
        trim_end=payload.end,
    )
    return {"ok": True, "video_path": str(trimmed), "duration": duration}


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
) -> dict:
    s = sm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    return s


@router.get("/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    sm: SessionManager = Depends(get_session_manager),
):
    """Server-Sent Events stream. Replaces polling for session + task updates.

    Events look like:  { "kind": "session"|"task"|"heartbeat", "data": {...}, "ts": ... }
    """
    if not sm.get_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")

    async def gen():
        # Emit a snapshot first so the client can render immediately.
        snap = sm.get_session(session_id)
        if snap:
            yield {"event": "session", "data": _json(snap)}
        for t in sm.list_tasks(session_id):
            yield {"event": "task", "data": _json(t)}

        async for evt in bus.subscribe(session_id):
            if await request.is_disconnected():
                break
            yield {"event": evt.kind, "data": evt.to_sse()}

    return EventSourceResponse(gen(), ping=15)


# ── Utilities ────────────────────────────────────────────────────────────────


def _resolve_video_path(session: dict[str, Any]) -> Path:
    vp = session.get("video_path") or ""
    return Path(vp)


def _json(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str)
