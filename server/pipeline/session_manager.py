"""
Thin re-export for back-compat. The real implementation now lives in
server/storage/db.py (SQLite-backed, event-bus wired). Pipeline tasks keep
importing `from .session_manager import SessionManager` and get the new store.
"""

from ..storage.db import SessionManager

__all__ = ["SessionManager"]
