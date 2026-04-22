"""
RunPod Serverless handler stub. Not used by the Pod deployment (which runs
uvicorn directly) — this is scaffolding for a later migration where we want
pay-per-second instead of a 24/7 Pod.

Serverless workers spin up on demand, receive one job payload, and exit. The
payload tells us what to do; we drive the same pipeline entry points that
the REST routes call.

Full Serverless spec: https://docs.runpod.io/serverless/handlers/overview

Example input:
    {
      "action": "analyze",                     # or "feature"
      "session_id": "abc123",
      "video_url": "https://.../clip.mp4",     # pre-signed URL or direct
      "feature": "heatmap"                     # required if action == "feature"
    }
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

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
        _sm = SessionManager(output_root=settings.output_root, db_path=settings.db_path)
    return _sm


def _download_video(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """Entry point called by the RunPod Serverless runtime."""
    payload = event.get("input", {}) or {}
    action = payload.get("action", "analyze")
    session_id = payload.get("session_id")
    if not session_id:
        return {"error": "session_id is required"}

    sm = _get_sm()

    try:
        s = sm.get_session(session_id)
        if not s and payload.get("video_url"):
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "video.mp4"
                _download_video(payload["video_url"], dest)
                final = settings.upload_root / session_id / "video.mp4"
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(final))
                sm.create_session(session_id, str(final))
                s = sm.get_session(session_id)

        if not s:
            return {"error": f"no session {session_id!r} and no video_url to bootstrap it"}

        if action == "analyze":
            pipeline_tasks.run_global_analysis(session_id, s, sm)
            return {"ok": True, "session": sm.get_session(session_id)}

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
