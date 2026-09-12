import hashlib

from app.schemas import ChatResponse

# In-memory cache
_cache: dict[str, ChatResponse] = {}

# Cache statistics
_cache_hits = 0
_cache_misses = 0


def create_cache_key(message: str) -> str:
    """
    Create a stable hash from the user's message.
    """
    normalized_message = message.strip().lower()

    return hashlib.sha256(normalized_message.encode("utf-8")).hexdigest()


def get_cached_response(message: str) -> ChatResponse | None:
    """
    Return cached response if available.
    """
    global _cache_hits, _cache_misses

    key = create_cache_key(message)

    if key in _cache:
        _cache_hits += 1
        return _cache[key]

    _cache_misses += 1
    return None


def set_cached_response(message: str, response: ChatResponse) -> None:
    """
    Store an LLM response in the cache.
    """
    key = create_cache_key(message)

    _cache[key] = response


def get_cache_stats() -> dict[str, int]:
    """
    Return current cache statistics.
    """
    return {
        "total_items": len(_cache),
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
    }
