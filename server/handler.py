"""
RunPod Serverless handler. Receives job payloads from the Vercel proxy and
drives the analysis pipeline.

Serverless workers spin up on demand, receive one job payload, and exit. The
payload tells us what to do; we drive the same pipeline entry points that
the REST routes call.

Full Serverless spec: https://docs.runpod.io/serverless/handlers/overview

Supported actions:
    "detect_frame" — Quick YOLO on a single frame (player selection UI)
    "track"        — SAMURAI tracking + full analysis
    "analyze"      — Full analysis (requires existing SAMURAI cache)
    "feature"      — Generate a single feature (heatmap, replay, etc.)

Example input:
    {
      "action": "analyze",
      "session_id": "abc123",
      "video_url": "https://.../clip.mp4",
      "feature": "heatmap"                     # required if action == "feature"
    }
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from supabase import create_client

from .config import settings
from .models.weights import ensure_weights
from .pipeline import tasks as pipeline_tasks
from .storage.db import SessionManager

log = logging.getLogger(__name__)

_sm: SessionManager | None = None


def _get_sm() -> SessionManager:
    global _sm
    if _sm is None:
        ensure_weights()
        _sm = SessionManager(output_root=settings.output_root)
    return _sm


def _supabase_storage_key_from_url(url: str) -> str | None:
    """
    Extract the storage object key from a Supabase Storage URL.

    Public URL shape:  https://<proj>.supabase.co/storage/v1/object/public/<bucket>/<key>
    Signed URL shape:  https://<proj>.supabase.co/storage/v1/object/sign/<bucket>/<key>?token=...
    Returns the <key> portion (everything after <bucket>/), or None if the URL
    doesn't look like Supabase Storage.
    """
    try:
        path = urlparse(url).path
        marker = "/storage/v1/object/"
        idx = path.find(marker)
        if idx == -1:
            return None
        # path tail: public/<bucket>/<key...>  or  sign/<bucket>/<key...>
        tail = path[idx + len(marker):]
        parts = tail.split("/", 2)
        if len(parts) < 3:
            return None
        return parts[2]
    except Exception:
        return None


def _download_video(url: str, dest: Path) -> None:
    """
    Download a video from Supabase Storage to local disk using the service-role
    SDK. This works whether the bucket is public or private — the service key
    bypasses RLS. Falls back to raw HTTP only if the URL isn't a recognizable
    Supabase Storage URL.
    """
    storage_key = _supabase_storage_key_from_url(url)
    if storage_key and settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
        supa = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        data = supa.storage.from_("videos").download(storage_key)
        dest.write_bytes(data)
        return

    # Fallback for non-Supabase URLs (e.g. R2, external CDN)
    import urllib.request
    with urllib.request.urlopen(url, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _ensure_local_video(session_id: str, video_url: str, sm: SessionManager) -> str:
    """
    Make sure the video exists on local disk. If not, download it from
    Supabase Storage. Returns the local path.
    """
    local_dir = settings.upload_root / session_id
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "video.mp4"

    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    if not video_url:
        raise ValueError(f"No video_url for session {session_id}")

    log.info("Downloading video from %s …", video_url[:80])
    _download_video(video_url, local_path)
    log.info("Downloaded %d MB", local_path.stat().st_size // (1024 * 1024))

    # Store the local path in the session so pipeline code can find it
    sm.update_status(session_id, "uploaded", video_path=str(local_path))
    return str(local_path)


def _run_auto_full_replay(session_id: str, sm: SessionManager) -> None:
    """Generate the showcase replay once analysis finishes, unless it exists."""
    existing = [
        t for t in sm.list_tasks(session_id)
        if t.get("task_type") == "full_replay"
        and t.get("status") in ("queued", "running", "done")
    ]
    if existing:
        return

    session = sm.get_session(session_id)
    if not session or session.get("status") != "analysis_done":
        return

    task_id = sm.create_task(session_id, "full_replay")
    log.info("[auto-full-replay] queued %s for session %s", task_id, session_id)
    pipeline_tasks.run_full_replay(session_id, session, task_id, sm)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """Entry point called by the RunPod Serverless runtime."""
    payload = event.get("input", {}) or {}
    action = payload.get("action", "analyze")
    session_id = payload.get("session_id")
    if not session_id:
        return {"error": "session_id is required"}

    video_url = payload.get("video_url", "")
    sm = _get_sm()

    try:
        # Ensure session exists in Supabase (frontend usually creates it)
        s = sm.get_session(session_id)
        if not s:
            if video_url:
                sm.create_session(session_id, video_url)
                s = sm.get_session(session_id)
            else:
                return {"error": f"no session {session_id!r} and no video_url"}

        # Make sure video is on local disk
        local_video = _ensure_local_video(session_id, video_url or s.get("video_url", ""), sm)
        # Refresh session to pick up updated video_path
        s = sm.get_session(session_id)

        # ── detect_frame: quick YOLO on a single frame ───────────────────
        if action == "detect_frame":
            frame_idx = int(payload.get("frame", 0))
            result = pipeline_tasks.detect_frame_players(session_id, s, frame_idx, sm)

            # Upload annotated frame to Supabase Storage so frontend can display it
            output_dir = sm.session_output_dir(session_id)
            frame_path = output_dir / result.get("annotated_frame_path", "first_frame.jpg")
            frame_url = None
            if frame_path.exists() and settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
                supa = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
                storage_key = f"{session_id}/first_frame.jpg"
                with open(frame_path, "rb") as f:
                    supa.storage.from_("videos").upload(
                        storage_key, f,
                        file_options={"content-type": "image/jpeg", "upsert": "true"}
                    )
                frame_url = supa.storage.from_("videos").get_public_url(storage_key)

            return {
                "players": result.get("players", []),
                "players_data": result.get("players", []),
                "annotated_frame_url": frame_url,
                "image_dimensions": result.get("image_dimensions"),
            }

        # ── track: SAMURAI tracking → full analysis → auto replay ────────
        if action == "track":
            bbox_raw = payload.get("bbox", {})
            frame = int(payload.get("frame", 0))

            # Convert xyxy bbox to xywh format expected by run_samurai_tracking
            if isinstance(bbox_raw, dict):
                x1 = float(bbox_raw.get("x1", 0))
                y1 = float(bbox_raw.get("y1", 0))
                x2 = float(bbox_raw.get("x2", 0))
                y2 = float(bbox_raw.get("y2", 0))
            elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox_raw]
            else:
                return {"error": "bbox must be {x1,y1,x2,y2} or [x1,y1,x2,y2]"}

            player_bbox = {
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
                "frame": frame,
            }
            s_merged = {
                **s,
                "selected_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "start_frame": frame,
            }

            # Phase 1: SAMURAI tracking
            pipeline_tasks.run_samurai_tracking(session_id, s_merged, player_bbox, sm)
            s_after = sm.get_session(session_id) or {}
            if s_after.get("status") != "tracking_done":
                return {"error": f"Tracking failed: {s_after.get('error', 'unknown')}"}

            # Phase 2: Global analysis
            pipeline_tasks.run_global_analysis(session_id, s_after, sm)

            # Phase 3: Auto-generate replay
            _run_auto_full_replay(session_id, sm)

            return {"ok": True, "session": sm.get_session(session_id)}

        # ── analyze: full analysis (skip SAMURAI) ────────────────────────
        if action == "analyze":
            pipeline_tasks.run_global_analysis(session_id, s, sm)
            _run_auto_full_replay(session_id, sm)
            return {"ok": True, "session": sm.get_session(session_id)}

        # ── feature: generate a single output (heatmap, speed chart…) ────
        if action == "feature":
            feature = payload.get("feature")
            from .routes.analysis import FEATURE_TASKS
            fn = FEATURE_TASKS.get(feature or "")
            if not fn:
                return {"error": f"unknown feature {feature!r}"}
            task_id = sm.create_task(session_id, feature)
            fn(session_id, s, task_id, sm)
            return {"ok": True, "task": sm.get_task(session_id, task_id)}

        return {"error": f"unknown action {action!r}"}
    except Exception as exc:
        log.exception("handler failed")
        return {"error": str(exc)}


# RunPod Serverless boilerplate (only imports when actually running serverless).
if __name__ == "__main__":
    try:
        import runpod  # type: ignore

        runpod.serverless.start({"handler": handler})
    except ImportError:
        print("runpod SDK not installed; this module is only meant to run inside RunPod Serverless.")
