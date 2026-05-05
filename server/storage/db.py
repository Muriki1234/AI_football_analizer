import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from supabase import create_client, Client

class SessionManager:
    """
    Supabase-backed session + task store.
    Directly syncs with the remote Postgres database.
    """

    def __init__(self, output_root: Path, db_path: Path | None = None):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment")
        
        self.client: Client = create_client(url, key)

    # ── Session CRUD ─────────────────────────────────────────────────────────

    def create_session(self, session_id: str, video_path: str) -> None:
        # Note: frontend usually creates the session, but we keep this for compatibility
        data = {
            "id": session_id,
            "video_url": video_path, # In serverless, this is the public URL from Supabase
            "status": "uploaded",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.client.table("sessions").upsert(data).execute()

    def get_session(self, session_id: str) -> dict | None:
        res = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if not res.data:
            return None
        # Map DB columns to what the pipeline expects
        session = res.data[0]
        session["session_id"] = session["id"]
        return session

    def update_status(
        self,
        session_id: str,
        status: str,
        progress: int | None = None,
        stage: str | None = None,
        error: str | None = None,
        **extra: Any,
    ) -> None:
        updates = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if progress is not None:
            updates["progress"] = progress
        if stage is not None:
            updates["stage"] = stage
        if error is not None:
            updates["error"] = error
        
        # In the new schema, we don't use extra_json, but if you want to store it,
        # you could add an 'extra' column to Supabase. For now, we skip it.

        self.client.table("sessions").update(updates).eq("id", session_id).execute()

    # ── Task CRUD ─────────────────────────────────────────────────────────────

    def create_task(self, session_id: str, task_type: str) -> str:
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        data = {
            "id": task_id,
            "session_id": session_id,
            "task_type": task_type,
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.client.table("tasks").insert(data).execute()
        return task_id

    def get_task(self, session_id: str, task_id: str) -> dict | None:
        res = self.client.table("tasks").select("*").eq("id", task_id).execute()
        return res.data[0] if res.data else None

    def list_tasks(self, session_id: str) -> list[dict]:
        res = self.client.table("tasks").select("*").eq("session_id", session_id).execute()
        return res.data

    def update_task(self, session_id: str, task_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        
        # Map internal keys to Supabase columns
        mapped = {}
        for k, v in kwargs.items():
            if k == "task_type": mapped["task_type"] = v
            elif k == "status": mapped["status"] = v
            elif k == "progress": mapped["progress"] = v
            elif k == "result": mapped["result"] = v
            elif k == "url": mapped["url"] = v
            elif k == "error": mapped["error"] = v
        
        if mapped:
            self.client.table("tasks").update(mapped).eq("id", task_id).execute()

    def session_output_dir(self, session_id: str) -> Path:
        p = self.output_root / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def delete_session(self, session_id: str) -> None:
        self.client.table("sessions").delete().eq("id", session_id).execute()

    def close(self) -> None:
        pass

__all__ = ["SessionManager"]
