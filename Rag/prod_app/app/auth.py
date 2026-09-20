"""API-key authentication dependency."""

import secrets

from fastapi import HTTPException, Request

from app.config import settings


def require_api_key(request: Request) -> str:
    """Read X-API-Key header, compare constant-time, return key id (truncated)."""
    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(key, settings.api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return f"key-{key[:6]}"