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


def _parse_match_periods(payload: dict, total_frames_hint: int) -> list[tuple[int, int]]:
    """
    Read match_periods from payload. Format from frontend:
      [[startFrame, endFrame], ...]  (sorted, non-overlapping)
    Defaults to a single full-video period if missing/empty.
    """
    raw = payload.get("match_periods")
    if not raw:
        return [(0, total_frames_hint)]
    out: list[tuple[int, int]] = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            s = max(0, int(p[0]))
            e = min(total_frames_hint, int(p[1]))
            if e > s + 1:
                out.append((s, e))
    if not out:
        return [(0, total_frames_hint)]
    out.sort()
    return out


def _parse_segments(payload: dict, total_frames_hint: int = 1500,
                    periods: list[tuple[int, int]] | None = None) -> list[dict]:
    """
    Normalize either:
      - single bbox (legacy): payload["bbox"] = {x1,y1,x2,y2} or [...]
      - segments array (new): payload["segments"] = [{frame, bbox, period_idx?}, ...]
    into the segments list shape that run_samurai_tracking_multi expects:
      [{"start_frame": int, "end_frame": int, "bbox": {x,y,w,h}, "period_idx": int}, ...]

    `periods` constrains each segment's end_frame to the bounds of its
    period — without this, the last seg in period 1 would span across
    a skipped halftime gap into period 2.
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
                "period_idx":  int(seg.get("period_idx", 0)),
                "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            })
        out.sort(key=lambda s_: s_["start_frame"])

        # Fill end_frame:
        #   - if periods supplied: end is min(next seg in same period start,
        #                                     this period's end_frame)
        #   - if no periods:       end is next seg start, or video end
        for i, s_ in enumerate(out):
            same_period_next = next(
                (n["start_frame"] for n in out[i + 1:]
                 if n["period_idx"] == s_["period_idx"]),
                None
            )
            if periods:
                period_end = periods[s_["period_idx"]][1] \
                    if s_["period_idx"] < len(periods) else total_frames_hint
                s_["end_frame"] = min(same_period_next or period_end, period_end)
            else:
                s_["end_frame"] = same_period_next or total_frames_hint
        return out

    # Legacy single-bbox path (kept for backward compatibility with the
    # single-segment FastAPI route still calling through run_samurai_tracking)
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
        "period_idx": 0,
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

    match_periods = _parse_match_periods(payload, total_frames_hint)
    segments = _parse_segments(payload, total_frames_hint=total_frames_hint,
                               periods=match_periods)
    if not segments:
        return {"error": "Provide either bbox+frame or segments=[{frame,bbox},...]"}

    # Enforce minimum period length (point E)
    MIN_PERIOD_FRAMES = 30 * 25  # 30s × ~25fps. Backend-side defensive check.
    for ps, pe in match_periods:
        if pe - ps < MIN_PERIOD_FRAMES:
            return {"error": f"Period {ps}..{pe} shorter than 30s minimum"}

    s_merged = {**s, "start_frame": segments[0]["start_frame"]}
    samurai_cache_path = str(sm.session_output_dir(session_id) / "samurai_tracking.pkl")
    sm.update_status(session_id, "tracking", progress=1,
                     stage="samurai_multi_pending",
                     samurai_cache_path=samurai_cache_path,
                     # Pin the periods on the session so downstream code
                     # (analysis, render) can read them without re-parsing
                     # the payload.
                     match_periods_frames=[list(p) for p in match_periods])

    s_merged = sm.get_session(session_id) or s_merged

    print(f"[TRACK] launching SAMURAI ({len(segments)} segment(s) across "
          f"{len(match_periods)} period(s)) || merged analysis in parallel")

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
        # Scale the thread.join() cap with video length too — 900s (15 min)
        # was fine for 30-min clips but broke 1.5h+ matches. Use 2× video
        # duration with a 15-min floor, computed from the same total_frames
        # we probed earlier.
        join_timeout = max(900.0, 2.0 * (total_frames_hint / 25.0))
        samurai_thread.join(timeout=join_timeout)

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
