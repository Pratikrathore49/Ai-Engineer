# app/redis_client.py
#
# Single shared Redis connection for the entire application.
# Every other module imports `redis_client` from here — never create
# a new Redis() connection anywhere else.
#
# Why one shared client?
#   redis-py's Redis() object uses a connection pool internally.
#   Creating one per request wastes sockets and slows things down.
#   One shared instance = one pool = efficient.

import logging
import os

import redis

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Read connection settings from environment variables                #
# ------------------------------------------------------------------ #
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB",   "0"))

# ------------------------------------------------------------------ #
#  Build the client                                                   #
# ------------------------------------------------------------------ #
# decode_responses=True  → Redis always stores bytes.
#   With this flag every GET returns a Python str, not b"string".
#   That means we can use normal string operations everywhere.
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,      # all responses come back as str, not bytes
    socket_connect_timeout=5,   # fail fast if Redis is not reachable
    socket_timeout=5,           # per-command timeout
)


def verify_redis_connection() -> None:
    """
    Ping Redis at startup.  If Redis is unreachable the app should
    refuse to start rather than silently running with broken features.

    Call this once from main.py lifespan before the server accepts traffic.
    """
    try:
        redis_client.ping()
        logger.info(
            "Redis connection established | host=%s | port=%d | db=%d",
            REDIS_HOST,
            REDIS_PORT,
            REDIS_DB,
        )
    except redis.ConnectionError as exc:
        # Log clearly so the operator knows exactly what failed
        logger.critical(
            "Redis connection FAILED | host=%s | port=%d | detail=%s",
            REDIS_HOST,
            REDIS_PORT,
            str(exc),
        )
        # Re-raise so FastAPI's lifespan propagates the error and the
        # process exits — fail fast is better than running degraded.
        raise
