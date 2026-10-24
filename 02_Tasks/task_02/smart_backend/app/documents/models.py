# app/models.py
#
# SQLAlchemy ORM models — one class = one database table.
# with a different model without to
# WHY ORM MODELS INSTEAD OF RAW SQL?
#   With raw SQL you'd write CREATE TABLE manually, handle types yourself,
#   and have no Python object to work with.
#   With SQLAlchemy ORM:
#     - Each class IS the table definition
#     - Each class attribute IS a column
#     - Python objects map directly to rows
#     - SQLAlchemy generates the SQL for you
#
# WHY THREE TABLES?
#   This is the standard RAG data model used in production:
#
#   documents          → one row per uploaded file (PDF, TXT, MD)
#       ↓ 1-to-many
#   document_chunks    → one row per text chunk cut from that file
#       ↓ 1-to-1
#   chunk_embeddings   → one row per chunk's vector (the numbers the
#                        embedding model produced)
#
#   Keeping embeddings in a separate table lets you re-embed chunks
#  uching the chunk text, and lets
#   you store multiple embeddings per chunk (different models).

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# ── Helper: generate a UUID string ────────────────────────────────
def _uuid() -> str:
    return str(uuid.uuid4())

# ── Table 1: documents ─────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=_uuid,
        # WHY UUID as primary key?
        #   Auto-increment integers are predictable (user can guess id=1, id=2).
        #   UUIDs are random and unguessable — safer for APIs.
        #   as_uuid=False → store as string, not Python UUID object,
        #   so json serialization works without extra steps.
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(
        String(10), nullable=False
        # WHY String(10)?
        #   "pdf", "txt", "md" — short strings, no need for Text.
        #   Constraining length catches bugs early.
    )
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="processing"
        # Values: "processing" | "ready" | "failed"
        # WHY a string and not an Enum?
        #   Simple strings are easier to extend later without a DB migration.
        #   For strict validation use Python Enum in the service layer.
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=True
        #Stores the full extracted text of the document.
        # nullable = True so old rows without content don't break.
        #task 17 (chunking) reads this column to split 
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        # WHY timezone=True?
        #   Always store timestamps in UTC. If you store local time you'll
        #   get bugs when servers are in different timezones.
    )

    # Relationship: one Document has many DocumentChunks
    # WHY define relationships?
    #   So you can write  document.chunks  in Python and SQLAlchemy
    #   automatically fetches the related rows. No manual JOIN needed.
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        # cascade="all, delete-orphan" means: if you delete a Document,
        # all its chunks (and their embeddings) are deleted automatically.
    )

# ── Table 2: document_chunks ───────────────────────────────────────

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        # WHY ForeignKey with ondelete="CASCADE"?
        #   Enforces referential integrity at the DATABASE level.
        #   If the parent Document is deleted, PostgreSQL automatically
        #   deletes all its chunks too — no orphaned rows.
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False
        # The position of this chunk in the original document.
        # chunk_index=0 is the first chunk, 1 is the second, etc.
        # Important for reconstructing document order when retrieving.
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=False
        # WHY Text and not String(n)?
        #   Text has no length limit — chunks can be hundreds of words.
        #   String(n) would silently truncate or error on long chunks.
    )
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=True
        # How many tokens this chunk uses. Useful for:
        #   - Staying under LLM context limits
        #   - Billing estimation
        #   - Chunking strategy tuning
        # Nullable because we might not always compute it.
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="chunks"
    )
    embedding: Mapped["ChunkEmbedding"] = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        uselist=False,
        # uselist=False → one-to-one relationship
        # Each chunk has exactly one embedding row.
    )

# ── Table 3: chunk_embeddings ──────────────────────────────────────

class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    chunk_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding = mapped_column(
        Vector(1536),
        nullable=False,
        # WHY Vector(1536)?
        #   1536 is the output dimension of OpenAI's text-embedding-3-small
        #   and text-embedding-ada-002 models — the most commonly used
        #   embedding models. Each chunk becomes a list of 1536 floats.
        #   pgvector stores this natively and can do similarity search on it.
        #
        # WHY NOT Mapped[list[float]]?
        #   pgvector's Vector type is not a standard Python type, so
        #   SQLAlchemy's Mapped[] annotation doesn't know about it.
        #   We use plain mapped_column() without Mapped[] here.
    )
    model: Mapped[str] = mapped_column(
        String(100), nullable=False
        # Which embedding model produced this vector.
        # e.g. "text-embedding-3-small", "text-embedding-ada-002"
        # Storing this lets you re-embed with a new model later
        # and compare results side by side.
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship back to the chunk
    chunk: Mapped["DocumentChunk"] = relationship(
        "DocumentChunk", back_populates="embedding"
    )
