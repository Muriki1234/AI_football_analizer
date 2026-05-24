import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from supabase import create_client, Client

# Keys that have their own dedicated columns in the 'sessions' table.
# Everything else goes into the 'extra' JSONB column.
_SESSION_COLUMNS = {
    "status", "progress", "stage", "error", "updated_at",
}

# Keys that have their own dedicated columns in the 'tasks' table.
_TASK_COLUMNS = {
    "task_type", "status", "progress", "result", "url", "error",
}


class SessionManager:
    """
    Supabase-backed session + task store.
    Directly syncs with the remote Postgres database.

    Pipeline workers store transient state (video_path, tracks_cache_path,
    samurai_cache_path, etc.) in the 'extra' JSONB column of the sessions
    table.  get_session() flattens that back into the dict so downstream
    code can do ``session.get("tracks_cache_path")`` transparently.
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
            "video_url": video_path,  # In serverless, this is the public URL from Supabase
            "status": "uploaded",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": json.dumps({"video_path": video_path}),
        }
        self.client.table("sessions").upsert(data).execute()

    def get_session(self, session_id: str) -> dict | None:
        res = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if not res.data:
            return None
        session = res.data[0]
        session["session_id"] = session["id"]
        # Flatten the extra JSONB column into the session dict so pipeline
        # code can transparently access e.g. session["tracks_cache_path"].
        extra_raw = session.pop("extra", None)
        if extra_raw:
            if isinstance(extra_raw, str):
                try:
                    extra_raw = json.loads(extra_raw)
                except (json.JSONDecodeError, TypeError):
                    extra_raw = {}
            if isinstance(extra_raw, dict):
                for k, v in extra_raw.items():
                    if k not in session:
                        session[k] = v
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
        """
        Atomic single-round-trip update.

        If `extra` kwargs are supplied, we call the Postgres function
        `merge_session_extra` which does `extra = extra || new_extra` inside
        the database — no SELECT-then-UPDATE, no race window between
        concurrent workers. Falls back to the legacy two-step path if the
        RPC isn't deployed yet (so existing instances keep working).
        """
        # Build the JSONB extra payload (None values explicitly delete keys
        # via the merge function — '{"key": null}' || existing produces null).
        extra_payload: dict[str, Any] = {}
        for k, v in extra.items():
            extra_payload[k] = v

        updates_payload = {"status": status}
        if progress is not None:
            updates_payload["progress"] = progress
        if stage is not None:
            updates_payload["stage"] = stage
        if error is not None:
            updates_payload["error"] = error

        # Fast path: one RPC call. Tries the SQL function added in
        # 20260525_jsonb_merge_rpc.sql.
        try:
            self.client.rpc(
                "merge_session_extra",
                {
                    "p_session_id": session_id,
                    "p_extra":      json.loads(json.dumps(extra_payload, default=str)),
                    "p_updates":    updates_payload,
                },
            ).execute()
            return
        except Exception as rpc_exc:
            # Likely the RPC wasn't deployed yet — fall through to legacy path.
            # Log once per cold start, not every call.
            if not getattr(self.__class__, "_rpc_warned", False):
                print(f"[db] merge_session_extra RPC unavailable ({rpc_exc!r}); "
                      f"falling back to two-step update")
                self.__class__._rpc_warned = True

        # ── Legacy fallback: SELECT-merge-UPDATE ─────────────────────────
        legacy_updates: dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if progress is not None:
            legacy_updates["progress"] = progress
        if stage is not None:
            legacy_updates["stage"] = stage
        if error is not None:
            legacy_updates["error"] = error

        if extra:
            current = self.client.table("sessions").select("extra").eq("id", session_id).execute()
            current_extra = {}
            if current.data:
                raw = current.data[0].get("extra")
                if raw:
                    if isinstance(raw, str):
                        try:
                            current_extra = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            current_extra = {}
                    elif isinstance(raw, dict):
                        current_extra = raw
            current_extra.update({k: v for k, v in extra.items() if v is not None})
            for k, v in extra.items():
                if v is None and k in current_extra:
                    current_extra[k] = None
            legacy_updates["extra"] = json.dumps(current_extra, default=str)

        self.client.table("sessions").update(legacy_updates).eq("id", session_id).execute()

    # ── Task CRUD ─────────────────────────────────────────────────────────────

    def create_task(self, session_id: str, task_type: str) -> str:
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        data = {
            "id": task_id,
            "session_id": session_id,
            "task_type": task_type,
            "status": "running",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
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

        mapped: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _TASK_COLUMNS:
                mapped[k] = v
            # Silently skip unknown keys (e.g. file_path, stage) so callers
            # don't crash.  If you need to persist them, add to _TASK_COLUMNS
            # and the Supabase table schema.

        if mapped:
            self.client.table("tasks").update(mapped).eq("id", task_id).execute()

    def clear_tasks(self, session_id: str) -> None:
        """Delete all tasks for a session (used before re-analysis)."""
        self.client.table("tasks").delete().eq("session_id", session_id).execute()

    # ── Cleanup helpers ──────────────────────────────────────────────────────

    def cleanup_zombies(self, timeout_minutes: int = 60) -> None:
        """Mark tasks stuck in 'running' for too long as failed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)).isoformat()
        # Find tasks that have been running for longer than the timeout
        res = (
            self.client.table("tasks")
            .select("id, session_id")
            .eq("status", "running")
            .lt("created_at", cutoff)
            .execute()
        )
        for task in (res.data or []):
            self.client.table("tasks").update({
                "status": "failed",
                "error": f"Timed out after {timeout_minutes} minutes",
            }).eq("id", task["id"]).execute()

    def expired_sessions(self, ttl_hours: int) -> list[str]:
        """Return IDs of sessions older than ttl_hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()
        res = (
            self.client.table("sessions")
            .select("id")
            .lt("created_at", cutoff)
            .execute()
        )
        return [row["id"] for row in (res.data or [])]

    # ── Utilities ────────────────────────────────────────────────────────────

    def session_output_dir(self, session_id: str) -> Path:
        p = self.output_root / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def delete_session(self, session_id: str) -> None:
        self.client.table("sessions").delete().eq("id", session_id).execute()

    def close(self) -> None:
        pass


__all__ = ["SessionManager"]
