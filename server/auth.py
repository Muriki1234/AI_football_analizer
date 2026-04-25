"""
Bearer token auth. The frontend sends X-API-Key; anything that mismatches gets 401.

If settings.API_KEY is empty, auth is disabled (dev mode). The startup log
warns loudly so this never silently ships to prod.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Query, status

from .config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        return  # dev mode — startup log has already warned
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


async def require_api_key_or_query(
    x_api_key: str | None = Header(default=None),
    key: str | None = Query(default=None),
) -> None:
    """
    Same as require_api_key but also accepts ?key= in the query string, so
    plain <img>/<video> tags can authenticate without custom headers. Use only
    on read-only endpoints (file serving, SSE).
    """
    if not settings.API_KEY:
        return
    candidate = x_api_key or key
    if not candidate or not hmac.compare_digest(candidate, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
