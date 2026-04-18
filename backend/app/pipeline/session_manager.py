"""
session_manager.py - Flask 版本（线程安全，无 async）
用内存字典管理 session 和任务状态，重启会丢失（够用于开发/测试）
生产环境可换成 Redis 或 SQLite
"""

import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class SessionManager:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self._sessions: dict[str, dict] = {}
        self._tasks: dict[str, dict[str, dict]] = {}
        # RLock allows re-entrant acquisition within the same thread
        # (simplifies callers that may hold the lock and call other guarded methods)
        self._lock = threading.RLock()

    # ── Session CRUD ─────────────────────────────────────────────────────────

    def create_session(self, session_id: str, video_path: str):
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "video_path": video_path,
                "status": "uploaded",          # uploaded → tracking → tracking_done
                                                # → analyzing → analysis_done
                "progress": 0,                 # 0-100
                "stage": "",                   # 细粒度阶段标签
                "error": None,
                "created_at": datetime.utcnow().isoformat(),
                # 由各阶段任务写入：
                "samurai_cache_path": None,
                "tracks_cache_path": None,
                "player_summary": None,
                "total_frames": None,
            }
            self._tasks[session_id] = {}

    def get_session(self, session_id: str) -> Optional[dict]:
        # Return a shallow copy under lock — prevents callers from reading
        # a half-updated session (e.g. status=analysis_done but player_summary not yet written)
        with self._lock:
            s = self._sessions.get(session_id)
            return dict(s) if s is not None else None

    def update_status(self, session_id: str, status: str,
                      progress: int = None, stage: str = None,
                      error: str = None, **extra):
        """
        更新 session 状态。extra kwargs 可以写入任意字段，
        例如 tracks_cache_path="...", player_summary={...}
        """
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return
            s["status"] = status
            if progress is not None:
                s["progress"] = progress
            if stage is not None:
                s["stage"] = stage
            if error is not None:
                s["error"] = error
            for k, v in extra.items():
                s[k] = v

    # ── Task CRUD ─────────────────────────────────────────────────────────────

    def create_task(self, session_id: str, task_type: str) -> str:
        """创建子任务，返回 task_id"""
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        with self._lock:
            if session_id not in self._tasks:
                self._tasks[session_id] = {}
            self._tasks[session_id][task_id] = {
                "task_id": task_id,
                "task_type": task_type,
                "status": "queued",    # queued → running → done | failed
                "progress": 0,
                "result": None,        # JSON 数据（如控球率数字）
                "file_path": None,     # 本地文件绝对路径
                "url": None,           # 前端可以直接 fetch 的相对 URL
                "error": None,
                "created_at": datetime.utcnow().isoformat(),
                "finished_at": None,
            }
        return task_id

    def get_task(self, session_id: str, task_id: str) -> Optional[dict]:
        with self._lock:
            t = self._tasks.get(session_id, {}).get(task_id)
            return dict(t) if t is not None else None

    def update_task(self, session_id: str, task_id: str, **kwargs):
        with self._lock:
            t = self._tasks.get(session_id, {}).get(task_id)
            if t:
                for k, v in kwargs.items():
                    t[k] = v
                if kwargs.get("status") in ("done", "failed"):
                    t["finished_at"] = datetime.utcnow().isoformat()

    def session_output_dir(self, session_id: str) -> Path:
        return self.output_root / session_id
