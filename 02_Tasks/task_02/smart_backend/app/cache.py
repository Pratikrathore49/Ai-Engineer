# app/cache.py
#
# Redis-backed response cache.
#
# WHY THIS IS BETTER THAN A DICT:
#   Old approach:  _cache = {}   → lives in RAM, dies on restart, not shared
#   New approach:  Redis GET/SET  → survives restarts, shared across instances,
#                                   auto-expires via TTL
#
# KEY DESIGN:
#   key   → "cache:{sha256_of_normalized_message}"
#   value → JSON-serialized ChatResponse
#   TTL   → 1 hour (3600 seconds) — stale answers auto-disappear
#
# WHY JSON?
#   Redis stores strings.  Python objects can't go in directly.
#   We serialize (dict → JSON string) on write, deserialize on read.

import hashlib
import json
import logging

import redis

from app.redis_client import redis_client
from app.schemas import ChatResponse

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3_600   # 1 hour


# ------------------------------------------------------------------ #
#  Helper: build the Redis key                                        #
# ------------------------------------------------------------------ #

def _make_cache_key(message: str) -> str:
    """
    Normalize the message and hash it so that:
      "Hello"  and  "  hello  "  hit the same cache entry.

    Prefix "cache:" keeps Redis keys namespaced — important when the
    same Redis DB holds rate_limit: and conversation: keys too.
    """
    normalized = message.strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"cache:{digest}"


# ------------------------------------------------------------------ #
#  Public interface (same signatures as the old dict-based cache)     #
# ------------------------------------------------------------------ #

def get_cached_response(message: str) -> ChatResponse | None:
    """
    Look up a cached response in Redis.

    Returns the deserialized ChatResponse if found, None on a miss
    or any Redis error (we never want a cache failure to crash a request).
    """
    key = _make_cache_key(message)

    try:
        raw = redis_client.get(key)          # returns None if key doesn't exist
    except redis.RedisError as exc:
        # Redis is unavailable → log and treat as a miss, keep serving
        logger.warning("Cache GET failed | key=%s | error=%s", key, exc)
        return None

    if raw is None:
        logger.debug("Cache miss | key=%s", key)
        return None

    try:
        data = json.loads(raw)               # JSON string → Python dict
        logger.info("Cache hit | key=%s", key)
        return ChatResponse(**data)          # dict → Pydantic model
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        # Corrupted value in Redis — treat as a miss, don't crash
        logger.warning("Cache deserialize error | key=%s | error=%s", key, exc)
        return None


def set_cached_response(message: str, response: ChatResponse) -> None:
    """
    Store a ChatResponse in Redis with a 1-hour TTL.

    `ex=CACHE_TTL_SECONDS` tells Redis to auto-delete the key after
    3600 seconds — no manual cleanup needed, ever.
    """
    key = _make_cache_key(message)

    try:
        payload = json.dumps(response.model_dump())   # Pydantic → dict → JSON str
        redis_client.set(key, payload, ex=CACHE_TTL_SECONDS)
        logger.debug("Cache set | key=%s | ttl=%ds", key, CACHE_TTL_SECONDS)
    except redis.RedisError as exc:
        # Cache write failure is non-fatal — the response still reaches the user
        logger.warning("Cache SET failed | key=%s | error=%s", key, exc)


def get_cache_stats() -> dict:
    """
    Return basic cache stats for the /cache/stats endpoint.

    DBSIZE counts ALL keys in the current Redis DB (cache + rate_limit +
    conversation).  For a dedicated cache DB (REDIS_DB=1) this is accurate.
    For a shared DB it's an over-count — production setups use separate DBs.
    """
    try:
        # Scan only keys that belong to the cache namespace
        cache_keys = list(redis_client.scan_iter("cache:*"))
        return {
            "total_cached_items": len(cache_keys),
            "ttl_seconds": CACHE_TTL_SECONDS,
            "backend": "redis",
        }
    except redis.RedisError as exc:
        logger.warning("Cache stats fetch failed | error=%s", exc)
        return {
            "total_cached_items": -1,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "backend": "redis",
        }
