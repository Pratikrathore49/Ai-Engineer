# app/rate_limiter.py
#
# Redis-backed rate limiter using INCR + EXPIRE.
#
# WHY THIS IS BETTER THAN A LIST OF TIMESTAMPS:
#   Old approach:  request_counts[api_key] = [t1, t2, ...]
#     → stored in RAM, resets on restart, broken under multiple server instances
#
#   New approach:  Redis INCR + EXPIRE
#     → atomic (no race condition), shared across instances, auto-resets
#
# HOW IT WORKS:
#   Every API key gets a Redis key:  "rate_limit:{api_key}"
#
#   On each request:
#     1. INCR  → atomically increments the counter (creates it at 0 if missing)
#     2. If counter == 1 → EXPIRE 60  (start the 60-second window)
#     3. If counter > 20 → return 429
#
#   After 60 seconds Redis deletes the key automatically → window resets.
#
# WHY ATOMIC?
#   INCR is a single Redis command.  Even with 100 simultaneous requests
#   each one gets a unique integer back — no two requests can read the same
#   count and both think they're under the limit.
#   With a Python dict you'd need a lock; with Redis you get it for free.

import logging
import os

import redis
from fastapi import Depends, HTTPException, Request

from app.auth import verify_api_key
from app.redis_client import redis_client

logger = logging.getLogger(__name__)

MAX_REQUESTS     = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
WINDOW_SECONDS   = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


def rate_limit(
    request: Request,
    api_key: str = Depends(verify_api_key),   # runs auth first, gives us the key
) -> None:
    """
    FastAPI dependency — raise 429 if the caller exceeds MAX_REQUESTS
    within WINDOW_SECONDS.

    Inject with:  Depends(rate_limit)
    """
    # Pull the already-validated key that auth stored on request.state
    key = f"rate_limit:{request.state.api_key}"
    # Safe hint for logs — never log the full key
    key_hint = request.state.api_key[:4] + "..."

    try:
        # Step 1 — atomically increment the counter
        # INCR creates the key with value 1 if it doesn't exist yet
        count = redis_client.incr(key)

        # Step 2 — on the very first request of this window, set the expiry.
        # We only do this when count == 1 to avoid resetting the window
        # on every request (which would let someone spam indefinitely).
        if count == 1:
            redis_client.expire(key, WINDOW_SECONDS)
            logger.debug(
                "Rate limit window started | key_hint=%s | window=%ds",
                key_hint,
                WINDOW_SECONDS,
            )

        # Step 3 — enforce the limit
        if count > MAX_REQUESTS:
            # Get remaining TTL so we can tell the client when to retry
            ttl = redis_client.ttl(key)
            logger.warning(
                "Rate limit exceeded | key_hint=%s | count=%d | limit=%d | ttl=%ds",
                key_hint,
                count,
                MAX_REQUESTS,
                ttl,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {ttl} seconds.",
                headers={"Retry-After": str(ttl)},
            )

        logger.debug(
            "Rate limit OK | key_hint=%s | count=%d/%d",
            key_hint,
            count,
            MAX_REQUESTS,
        )

    except redis.RedisError as exc:
        # If Redis is down we have two choices:
        #   a) block all requests (safe but breaks the API)
        #   b) allow through and log loudly (keeps the API up)
        # We choose (b) with a loud warning so ops can react quickly.
        logger.error(
            "Rate limiter Redis error | key_hint=%s | error=%s | ALLOWING REQUEST",
            key_hint,
            exc,
        )
        # Do not re-raise — fall through and allow the request
