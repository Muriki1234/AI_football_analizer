"""
FastAPI dependency providers. The SessionManager is a singleton attached to
the app state in main.py's lifespan — these helpers pull it out for routes.
"""

from __future__ import annotations

from fastapi import Request

from .storage.db import SessionManager
from .workers.pool import WorkerPool


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager  # type: ignore[no-any-return]


def get_worker_pool(request: Request) -> WorkerPool:
    return request.app.state.worker_pool  # type: ignore[no-any-return]
