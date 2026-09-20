"""Tiny TTL cache without external deps.

Key = query string, value = cached response. TTL from settings.
"""

import threading
import time

from app.config import settings


class TTLCache:
    def __init__(self, ttl_seconds: int = settings.cache_ttl_seconds, max_size: int = 2000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            if len(self._store) >= self.max_size:
                # drop oldest by timestamp (simple approximation)
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


answer_cache = TTLCache()