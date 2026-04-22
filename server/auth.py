"""
Bearer token auth. The frontend sends X-API-Key; anything that mismatches gets 401.

If settings.API_KEY is empty, auth is disabled (dev mode). The startup log
warns loudly so this never silently ships to prod.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.API_KEY:
        return  # dev mode — startup log has already warned
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
