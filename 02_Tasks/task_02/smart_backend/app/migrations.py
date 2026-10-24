# app/migrations.py
#
# Creates all database tables on startup.
#
# WHY NOT ALEMBIC YET?
#   Alembic is the proper migration tool for production — it tracks
#   schema changes over time, like git for your database.
#   But it has setup overhead. For now, create_all() is enough:
#     - If tables don't exist → creates them
#     - If tables already exist → does nothing (safe to run every startup)
#   When we add Alembic later, we'll replace this file.
#
# WHY A SEPARATE FILE AND NOT JUST db.py?
#   db.py is about the connection (engine, sessions).
#   migrations.py is about the schema (tables, structure).
#   Separation of concerns — each file has one job.

import logging
from sqlalchemy import text

from app.db import Base, get_engine
# Importing models is REQUIRED even though we don't use them directly.
# WHY? Base.metadata only knows about a model after that model's class
# has been defined (i.e., its module imported).
# If we don't import models here, Base.metadata.create_all() would
# create zero tables because it doesn't know they exist yet.
from app.documents.models import ChunkEmbedding, Document, DocumentChunk  # noqa: F401

logger = logging.getLogger(__name__)

def run_migrations() -> None:
    """
    Create all tables defined in models.py if they don't already exist.

    Safe to call on every startup — create_all() checks existence first.
    Called from main.py lifespan after verify_db_connection().
    """
    try:
        engine = get_engine()

        # Base.metadata holds every table definition from every model
        # that inherits from Base. create_all() issues CREATE TABLE IF
        # NOT EXISTS for each one.
        Base.metadata.create_all(bind=engine)
    # Add content column to documents if it doesn't exist yet.
        # WHY raw SQL here?
        #   create_all() never alters existing tables — it only creates
        #   missing ones. For adding a column to an existing table we need
        #   ALTER TABLE. "IF NOT EXISTS" makes it safe to run every startup.
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content TEXT"
            ))
            conn.commit()
        logger.info(
            "Database migrations complete | tables=%s",
            list(Base.metadata.tables.keys()),
        )

    except Exception as exc:
        logger.critical(
            "Database migration FAILED | detail=%s", str(exc)
        )
        raise
