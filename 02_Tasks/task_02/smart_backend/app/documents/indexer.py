# app/documents/indexer.py
#
# The indexing pipeline — runs as a background task after upload.
#
# FLOW:
#   1. Load the document's extracted text from the DB
#   2. Chunk the text into overlapping word windows
#   3. Embed all chunks in one API batch call
#   4. Save DocumentChunk rows to document_chunks table
#   5. Save ChunkEmbedding rows to chunk_embeddings table
#   6. Update document.status → "ready"
#
# If anything fails, document.status → "failed" so the user
# can see something went wrong via GET /documents/{id}.
#
# WHY DOES THIS RUN AS A BACKGROUND TASK?
#   Chunking + embedding takes 2-10 seconds depending on document size.
#   Running it inline would block the HTTP response — bad UX.
#   FastAPI's BackgroundTasks runs this function AFTER the response
#   is already sent to the client. The client gets status="processing"
#   immediately, then polls GET /documents/{id} to see when it's "ready".

import logging
import uuid
from datetime import datetime, timezone

from app.core.db import get_session_local
from app.documents.chunker import chunk_text
from app.documents.embedder import embed_texts, EMBEDDING_MODEL
from app.documents.models import ChunkEmbedding, Document, DocumentChunk
logger = logging.getLogger(__name__)

def index_document(document_id: str) -> None:
    """
    Full indexing pipeline for one document.

    Creates its own DB session — the request session is closed by FastAPI
    before this background task runs, so we never reuse it.

    Side effects:
        - Inserts rows into document_chunks
        - Inserts rows into chunk_embeddings
        - Updates documents.status to "ready" or "failed"
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    logger.info("Indexing started | document_id=%s", document_id)

    # We wrap everything in try/except so a failure updates status="failed"
    # instead of leaving the document stuck at status="processing" forever.
    try:
        # ── Step 1: Load document from DB ─────────────────────────
        document = db.query(Document).filter(Document.id == document_id).first()

        if not document:
            logger.error(
                "Indexing failed — document not found | document_id=%s",
                document_id,
            )
            return

        if not document.content:
            logger.error(
                "Indexing failed — document has no content | document_id=%s",
                document_id,
            )
            _mark_failed(db, document)
            return

        # ── Step 2: Chunk the text ─────────────────────────────────
        chunks = chunk_text(document.content)

        if not chunks:
            logger.warning(
                "No chunks produced | document_id=%s | content_length=%d",
                document_id, len(document.content),
            )
            _mark_failed(db, document)
            return

        logger.info(
            "Chunking complete | document_id=%s | chunk_count=%d",
            document_id, len(chunks),
        )

        # ── Step 3: Embed all chunks in one batch API call ─────────
        # embed_texts sends all chunks together — one HTTP call, not N.
        vectors = embed_texts(chunks)

        # Sanity check: we must get back exactly one vector per chunk.
        # If these don't match something went wrong in the API call.
        if len(vectors) != len(chunks):
            logger.error(
                "Vector count mismatch | chunks=%d | vectors=%d | document_id=%s",
                len(chunks), len(vectors), document_id,
            )
            _mark_failed(db, document)
            return

        logger.info(
            "Embeddings received | document_id=%s | vector_count=%d | dimensions=%d",
            document_id, len(vectors), len(vectors[0]),
        )

        # ── Step 4 + 5: Save chunks and embeddings to DB ──────────
        now = datetime.now(timezone.utc)

        for index, (chunk_text_content, vector) in enumerate(zip(chunks, vectors)):
            # Create DocumentChunk row
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=index,
                content=chunk_text_content,
                token_count=len(chunk_text_content.split()),
                # WHY use word count as token_count?
                #   Real token count requires a tokenizer (tiktoken).
                #   Word count is a close enough approximation for now.
                #   We'll improve this when we add tiktoken later.
                created_at=now,
            )
            db.add(chunk)
            db.flush()
            # WHY flush() before creating the embedding?
            #   flush() sends the INSERT to PostgreSQL within the current
            #   transaction but doesn't commit yet. This assigns the chunk
            #   its DB row so ChunkEmbedding's foreign key (chunk_id) is valid.
            #   Without flush(), chunk.id exists in Python but not in the DB yet,
            #   and the FK constraint would fail.

            # Create ChunkEmbedding row
            embedding = ChunkEmbedding(
                id=str(uuid.uuid4()),
                chunk_id=chunk.id,
                embedding=vector,
                model=EMBEDDING_MODEL,
                created_at=now,
            )
            db.add(embedding)

        # ── Step 6: Commit everything and mark document ready ──────
        document.status = "ready"
        db.commit()

        logger.info(
            "Indexing complete | document_id=%s | chunks=%d | status=ready",
            document_id, len(chunks),
        )

    except Exception as exc:
        logger.error(
            "Indexing failed | document_id=%s | error_type=%s | detail=%s",
            document_id, type(exc).__name__, str(exc),
            exc_info=True,
        )
        # Try to mark the document as failed so it doesn't stay "processing"
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                _mark_failed(db, document)
        except Exception:
            db.rollback()
    finally:
        db.close()

def _mark_failed(db, document: Document) -> None:
    """Set document status to failed and commit."""
    document.status = "failed"
    db.commit()
    logger.info("Document marked as failed | document_id=%s", document.id)
