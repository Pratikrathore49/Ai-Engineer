# app/main.py
# Entry point for task_02 — production Redis backend.
# Key difference from task_01: lifespan now calls verify_redis_connection()
# so the server refuses to start if Redis is unreachable.

from dotenv import load_dotenv
# Prefer this project's configuration over stale variables inherited from
# an older local Docker/PostgreSQL setup.
load_dotenv(override=True)

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
from app.db import verify_db_connection
from app.redis_client import verify_redis_connection
from app.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        verify_redis_connection()
    except redis.ConnectionError:
        logger.critical(
            "Startup aborted - Redis is not reachable. "
            "Run: docker compose up -d"
        )
        raise
    try:
        verify_db_connection()
    except Exception:
        logger.critical(
            "Startup aborted - PostgreSQL is not reachable. "
            "Run: docker compose up -d"
        )
        raise

    logger.info(
        "Smart Reply API v2 started | backend=Redis+Postgresql | ready"
    )
    yield
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
