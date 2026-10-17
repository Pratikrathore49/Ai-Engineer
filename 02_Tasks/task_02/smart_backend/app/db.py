# app/db.py
#
# SQLAlchemy database engine, session factory, and startup health check.
#
# KEY FIX vs previous version:
#   The engine is built LAZILY inside get_engine() instead of at module
#   import time. This guarantees load_dotenv() in main.py has already
#   run before os.getenv() is called here.
#
#   Old (broken) pattern:
#       DATABASE_URL = f"...{os.getenv('POSTGRES_PASSWORD')}..."  ← runs at import
#       engine = create_engine(DATABASE_URL)                       ← wrong values
#
#   New (correct) pattern:
#       def get_engine():
#           url = f"...{os.getenv('POSTGRES_PASSWORD')}..."       ← runs at call time
#           return create_engine(url)                              ← correct values

import logging
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# Module-level engine reference — populated on first call to get_engine()
_engine = None


def get_engine():
    """
    Build (or return the cached) SQLAlchemy engine.

    Called lazily so that os.getenv() runs AFTER load_dotenv() has
    populated the environment from .env.
    """
    global _engine
    if _engine is not None:
        return _engine

    host     = os.getenv("POSTGRES_HOST",     "127.0.0.1")
    port     = os.getenv("POSTGRES_PORT",     "5432")
    db       = os.getenv("POSTGRES_DB",       "knowledgepilot")
    user     = os.getenv("POSTGRES_USER",     "kp_user")
    password = os.getenv("POSTGRES_PASSWORD", "kp_password")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    logger.debug("Building SQLAlchemy engine | host=%s | port=%s | db=%s", host, port, db)

    _engine = create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,   # discard stale connections automatically
        echo=False,
    )
    return _engine


def get_session_local():
    """Return a sessionmaker bound to the engine."""
    return sessionmaker(
        bind=get_engine(),
        autocommit=False,
        autoflush=False,
    )


# ── Base class for future ORM models ──────────────────────────────

class Base(DeclarativeBase):
    pass


# ── FastAPI dependency ─────────────────────────────────────────────

def get_db():
    """
    Yield a database session for the duration of one request.
    The session is always closed in the finally block — even on error.

    Usage in a route:
        @router.get("/something")
        def my_route(db: Session = Depends(get_db)):
            result = db.execute(text("SELECT 1"))
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Startup health check ───────────────────────────────────────────

def verify_db_connection() -> None:
    """
    Verify PostgreSQL is reachable and enable pgvector extension.
    Called once from main.py lifespan. Raises on failure (fail fast).
    """
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB",   "knowledgepilot")

    try:
        engine = get_engine()

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        logger.info(
            "Database connection established | host=%s | port=%s | db=%s",
            host, port, db,
        )
        logger.info("pgvector extension enabled")

    except Exception as exc:
        logger.critical(
            "Database connection FAILED | host=%s | port=%s | detail=%s",
            host, port, str(exc),
        )
        raise
