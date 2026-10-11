# app/main.py
# Entry point for task_02 — production Redis backend.
# Key difference from task_01: lifespan now calls verify_redis_connection()
# so the server refuses to start if Redis is unreachable.

from dotenv import load_dotenv
load_dotenv()

from app.logger import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

from app.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.health import health_check
from app.redis_client import verify_redis_connection
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────
    # Fail fast: if Redis is down the server exits here with a clear error
    # instead of silently starting and failing on the first request.
    try:
        verify_redis_connection()
    except redis.ConnectionError:
        logger.critical(
            "Startup aborted — Redis is not reachable. "
            "Run: docker run -d --name redis-dev -p 6379:6379 redis:7-alpine"
        )
        raise   # re-raise causes uvicorn to exit non-zero

    logger.info("Smart Reply API v2 started | backend=Redis | ready to accept requests")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Smart Reply API shutting down")


app = FastAPI(
    title="Smart Reply API",
    version="2.0.0",
    description="Production-grade FastAPI + Redis backend",
    lifespan=lifespan,
)

# Register exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/")
def root():
    return {"message": "Smart Reply API v2 is running", "backend": "Redis"}


@app.get("/health")
def health():
    return health_check()


app.include_router(router)
