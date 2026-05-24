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
_supabase_client = None   # module-level cache — see _get_supabase()


def _get_supabase():
    """
    Cached service-role Supabase client. Building one isn't expensive, but
    we were calling create_client() once per video download AND once per
    annotated-frame upload, which adds up to 2 fresh clients per analysis.
    Module-level singleton kills that.
    """
    global _supabase_client
    if _supabase_client is None:
        if not (settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY):
            return None
        _supabase_client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY
        )
    return _supabase_client


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


_DOWNLOAD_RETRIES = 4
_DOWNLOAD_BACKOFF_BASE = 2.0   # 2s, 4s, 8s, 16s


def _download_video(url: str, dest: Path) -> None:
    """
    Download a video from Supabase Storage to local disk using the service-role
    SDK. This works whether the bucket is public or private — the service key
    bypasses RLS. Falls back to raw HTTP only if the URL isn't a recognizable
    Supabase Storage URL.

    Retries on any error with exponential backoff (4 attempts: 2/4/8/16 s).
    For large videos over flaky RunPod networking this is essential — a
    single TCP reset previously failed the whole pipeline.
    """
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(_DOWNLOAD_RETRIES):
        try:
            _download_video_once(url, dest)
            # Sanity check: empty / truncated file should also retry
            size = dest.stat().st_size if dest.exists() else 0
            if size < 1024:
                raise RuntimeError(f"downloaded file too small ({size} B)")
            if attempt > 0:
                log.info("Video download succeeded on attempt %d", attempt + 1)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < _DOWNLOAD_RETRIES - 1:
                wait = _DOWNLOAD_BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Video download attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, _DOWNLOAD_RETRIES, exc, wait,
                )
                _time.sleep(wait)
                # Wipe partial file before retry to avoid corrupted state
                try:
                    if dest.exists():
                        dest.unlink()
                except Exception:
                    pass
            else:
                log.error(
                    "Video download exhausted all %d attempts: %s",
                    _DOWNLOAD_RETRIES, exc,
                )
    raise RuntimeError(
        f"Failed to download {url[:80]}… after {_DOWNLOAD_RETRIES} attempts: {last_exc}"
    )


def _download_video_once(url: str, dest: Path) -> None:
    """Single download attempt — the retry loop is in _download_video()."""
    storage_key = _supabase_storage_key_from_url(url)
    supa = _get_supabase()
    if storage_key and supa is not None:
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


def _action_detect_frame(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """Quick YOLO on a single frame for the player-selection UI."""
    frame_idx = int(payload.get("frame", 0))
    result = pipeline_tasks.detect_frame_players(session_id, s, frame_idx, sm)

    output_dir = sm.session_output_dir(session_id)
    frame_path = output_dir / result.get("annotated_frame_path", "first_frame.jpg")
    frame_url = None
    supa = _get_supabase()
    if frame_path.exists() and supa is not None:
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


def _parse_segments(payload: dict, total_frames_hint: int = 1500) -> list[dict]:
    """
    Normalize either:
      - single bbox (legacy): payload["bbox"] = {x1,y1,x2,y2} or [...]
      - segments array (new): payload["segments"] = [{frame, bbox}, ...]
    into the segments list shape that run_samurai_tracking_multi expects:
      [{"start_frame": int, "end_frame": int, "bbox": {x,y,w,h}}, ...]
    """
    segments_in = payload.get("segments")
    if segments_in:
        out = []
        for seg in segments_in:
            bb = seg.get("bbox", {})
            if isinstance(bb, dict):
                x1 = float(bb.get("x1", bb.get("x", 0)))
                y1 = float(bb.get("y1", bb.get("y", 0)))
                x2 = float(bb.get("x2", bb.get("x", 0) + bb.get("w", 0)))
                y2 = float(bb.get("y2", bb.get("y", 0) + bb.get("h", 0)))
            elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                x1, y1, x2, y2 = [float(v) for v in bb]
            else:
                continue
            out.append({
                "start_frame": int(seg.get("frame", 0)),
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            })
        out.sort(key=lambda s_: s_["start_frame"])
        # Fill in end_frame from the next segment's start, last one ends at video end
        for i, s_ in enumerate(out):
            s_["end_frame"] = (out[i + 1]["start_frame"]
                                if i + 1 < len(out) else total_frames_hint)
        return out

    # Legacy single-bbox path
    bbox_raw = payload.get("bbox", {})
    frame = int(payload.get("frame", 0))
    if isinstance(bbox_raw, dict):
        x1 = float(bbox_raw.get("x1", 0)); y1 = float(bbox_raw.get("y1", 0))
        x2 = float(bbox_raw.get("x2", 0)); y2 = float(bbox_raw.get("y2", 0))
    elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox_raw]
    else:
        return []
    return [{
        "start_frame": frame,
        "end_frame": total_frames_hint,
        "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
    }]


def _action_track(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """
    SAMURAI tracking (multi-segment, parallel) + global analysis (concurrent
    on main thread) → auto-generate replay.

    The two GPU jobs run in parallel — SAMURAI on extracted JPGs via
    subprocesses, the analysis on the original video via the main process.
    They only sync up right before _compute_player_summary (which needs both).
    """
    import threading

    # Try to probe total_frames so we can default the last segment's end_frame.
    total_frames_hint = 1500
    video_path_local = s.get("video_path") or ""
    if video_path_local:
        try:
            import cv2 as _cv2  # noqa: WPS433 (local import for cold-start speed)
            cap = _cv2.VideoCapture(video_path_local)
            t = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if t > 0:
                total_frames_hint = t
        except Exception:
            pass

    segments = _parse_segments(payload, total_frames_hint=total_frames_hint)
    if not segments:
        return {"error": "Provide either bbox+frame or segments=[{frame,bbox},...]"}

    s_merged = {**s, "start_frame": segments[0]["start_frame"]}
    # Pre-write the expected samurai cache path so run_global_analysis can
    # find it later (it polls for the file's existence).
    samurai_cache_path = str(sm.session_output_dir(session_id) / "samurai_tracking.pkl")
    sm.update_status(session_id, "tracking", progress=1,
                     stage="samurai_multi_pending",
                     samurai_cache_path=samurai_cache_path)

    # Refresh session so it includes samurai_cache_path
    s_merged = sm.get_session(session_id) or s_merged

    print(f"[TRACK] launching SAMURAI ({len(segments)} segment(s)) || "
          f"merged analysis in parallel")

    # Event-based handoff between SAMURAI thread and analysis thread.
    # Replaces the old filesystem busy-poll (sleep 1s in a loop): zero CPU
    # while waiting, instant wake when SAMURAI finishes, and no RunPod
    # seconds wasted spinning.
    samurai_done = threading.Event()
    samurai_err: dict = {}

    def _samurai_worker():
        try:
            pipeline_tasks.run_samurai_tracking_multi(
                session_id, s_merged, segments, sm
            )
        except Exception as e:
            samurai_err["exc"] = e
            log.exception("SAMURAI multi-segment failed")
        finally:
            samurai_done.set()

    samurai_thread = threading.Thread(target=_samurai_worker, daemon=True)
    samurai_thread.start()

    # Run analysis on this thread, concurrently with SAMURAI subprocesses.
    # run_global_analysis blocks on `samurai_done` (passed via attribute on
    # session dict) right before the summary step.
    s_merged["_samurai_done_event"] = samurai_done
    try:
        pipeline_tasks.run_global_analysis(session_id, s_merged, sm)
    finally:
        samurai_thread.join(timeout=900)

    if samurai_err:
        return {"error": f"SAMURAI failed: {samurai_err['exc']}"}

    _run_auto_full_replay(session_id, sm)
    return {"ok": True, "session": sm.get_session(session_id)}


def _action_analyze(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """Full analysis (assumes SAMURAI cache already present)."""
    pipeline_tasks.run_global_analysis(session_id, s, sm)
    _run_auto_full_replay(session_id, sm)
    return {"ok": True, "session": sm.get_session(session_id)}


def _action_feature(session_id: str, s: dict, payload: dict, sm: SessionManager) -> dict:
    """Generate a single feature output (heatmap, speed chart, etc.)."""
    feature = payload.get("feature")
    from .routes.analysis import FEATURE_TASKS
    fn = FEATURE_TASKS.get(feature or "")
    if not fn:
        return {"error": f"unknown feature {feature!r}"}
    task_id = sm.create_task(session_id, feature)
    fn(session_id, s, task_id, sm)
    return {"ok": True, "task": sm.get_task(session_id, task_id)}


# Adding a new action = one entry here, no main-handler changes needed.
ACTIONS = {
    "detect_frame": _action_detect_frame,
    "track":        _action_track,
    "analyze":      _action_analyze,
    "feature":      _action_feature,
}


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """Entry point called by the RunPod Serverless runtime."""
    payload = event.get("input", {}) or {}
    action = payload.get("action", "analyze")
    session_id = payload.get("session_id")
    if not session_id:
        return {"error": "session_id is required"}

    fn = ACTIONS.get(action)
    if not fn:
        return {"error": f"unknown action {action!r}"}

    video_url = payload.get("video_url", "")
    sm = _get_sm()

    try:
        s = sm.get_session(session_id)
        if not s:
            if video_url:
                sm.create_session(session_id, video_url)
                s = sm.get_session(session_id)
            else:
                return {"error": f"no session {session_id!r} and no video_url"}

        _ensure_local_video(session_id, video_url or s.get("video_url", ""), sm)
        s = sm.get_session(session_id)

        return fn(session_id, s, payload, sm)
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
