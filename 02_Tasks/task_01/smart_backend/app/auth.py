# app/auth.py

import logging
import os

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)

VALID_API_KEYS = {
    key.strip()
    for key in os.getenv("SECURITY_API_KEYS", "").split(",")
    if key.strip()
}
if not VALID_API_KEYS:
    logger.warning("No API key configured - all requests will be rejected")

def verify_api_key(
    request: Request,
    api_key: str | None = Header(
        default=None,
        alias="X-API-Key"
    )
):
    client_ip = request.client.host if request.client else "unknown"

    if api_key is None:
        logger.warning(
            "Auth failure | ip=%s | reason=missing X-API-Key header",
            client_ip
        )
        raise HTTPException(status_code=401, detail="X-API-Key header is required")

    if api_key not in VALID_API_KEYS:
        logger.warning(
            "Auth failure | ip=%s | reason=invalid API key",
            client_ip
        )
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.debug("Auth success | ip=%s", client_ip)
    return api_key
