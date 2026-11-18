# app/documents/retriever.py
#
# Semantic retrieval — find the most relevant chunks for a query.
#
# THE CORE IDEA:
#   When we indexed a document we stored each chunk's meaning as a
#   vector (list of 1536 floats) in the chunk_embeddings table.
#   Now when a user asks a question, we embed their question using
#   the SAME model — producing a vector in the SAME space.
#   pgvector finds the stored vectors closest to the query vector.
#   Close vectors = similar meaning = relevant chunks.
#
# WHY DO THIS IN SQL AND NOT PYTHON?
#   The naive approach: load all chunk vectors into Python, compute
#   cosine similarity for each one, sort, return top K.
#   The problem: if you have 100,000 chunks, you load 100,000 vectors
#   into RAM on every single search query. That's ~600MB of data per
#   request and very slow.
#   pgvector does the distance computation inside PostgreSQL on the
#   stored data — no data transfer until we know the top K results.
#   This is the entire point of a vector database.
#
# THE SQL QUERY WE'RE BUILDING:
#   SELECT
#       dc.id,
#       dc.content,
#       dc.chunk_index,
#       d.filename,
#       1 - (ce.embedding <=> :query_vector) AS similarity_score
#   FROM chunk_embeddings ce
#   JOIN document_chunks dc ON dc.id = ce.chunk_id
#   JOIN documents d ON d.id = dc.document_id
#   WHERE d.status = 'ready'
#   [AND d.id = :document_id]        ← optional filter
#   ORDER BY ce.embedding <=> :query_vector   ← closest first
#   LIMIT :top_k
#
# WHY THREE TABLES IN ONE QUERY?
#   chunk_embeddings → has the vector (for distance math)
#   document_chunks  → has the text content (what we return)
#   documents        → has the filename (so user knows which file)
#   They're separate tables so we JOIN them to get everything in one shot.

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.documents.embedder import embed_texts
from app.documents.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

def retrieve_chunks(
    query: str,
    db: Session,
    document_id: str | None = None,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
) -> list[RetrievedChunk]:
    """
    Find the most semantically relevant chunks for a query.

    Args:
        query:               the user's question
        db:                  SQLAlchemy session
        document_id:         if provided, search only this document.
                             if None, search all ready documents.
        top_k:               max number of results to return
        similarity_threshold: minimum similarity score (0-1).
                             chunks below this score are filtered out.
                             0.3 means "at least 30% similar" —
                             filters out completely unrelated results.

    Returns:
        List of RetrievedChunk ordered by similarity (highest first).
    """

    # ── Step 1: Embed the query ────────────────────────────────────
    # We use the exact same embed_texts() function used during indexing.
    # WHY the same function?
    #   Both the query and the chunks must live in the SAME vector space.
    #   If you indexed with model A but searched with model B, the vectors
    #   would be incompatible — like comparing temperatures in Celsius vs
    #   Fahrenheit. Same model = same space = meaningful comparison.
    logger.info("Retrieval started | query_preview='%s'", query[:50])

    query_vectors = embed_texts([query])

    if not query_vectors:
        logger.error("Failed to embed query | query='%s'", query[:50])
        return []

    query_vector = query_vectors[0]
    # query_vector is now a list like [0.12, -0.43, 0.87, ...]
    # It has the same number of dimensions as the stored chunk vectors.

    # ── Step 2: Build and run the pgvector SQL query ───────────────
    # We use raw SQL with text() here because SQLAlchemy's ORM doesn't
    # have native support for pgvector's <=> operator.
    # text() lets us write raw SQL while still using the managed session
    # (connection pooling, transaction management still work).
    #
    # The :param syntax is a BIND PARAMETER — SQLAlchemy substitutes the
    # actual value safely. This prevents SQL injection.
    #
    # WHY cast(query_vector as string)?
    #   pgvector expects the vector as a string like '[0.1, 0.2, ...]'
    #   Python's list str() produces exactly that format.

    # Build optional document filter clause
    document_filter = ""
    params: dict = {
        "query_vector": str(query_vector),
        "top_k": top_k,
    }

    if document_id:
        document_filter = "AND d.id = :document_id"
        params["document_id"] = document_id
        logger.debug("Filtering by document_id=%s", document_id)

    sql = text(f"""
        SELECT
            dc.id          AS chunk_id,
            dc.content     AS content,
            dc.chunk_index AS chunk_index,
            d.filename     AS filename,
            1 - (ce.embedding <=> CAST(:query_vector AS vector)) AS similarity_score
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON dc.id = ce.chunk_id
        JOIN documents d        ON d.id  = dc.document_id
        WHERE d.status = 'ready'
        {document_filter}
        ORDER BY ce.embedding <=> CAST(:query_vector AS vector)
        LIMIT :top_k
    """)
    # WHY ORDER BY distance and not ORDER BY similarity_score DESC?
    #   They produce the same ranking — distance and similarity are
    #   inverse of each other. Ordering by distance (ASC) is slightly
    #   more efficient because pgvector uses the raw distance internally.
    #   We compute similarity_score only in the SELECT for the response.

    logger.debug("Running pgvector similarity search | top_k=%d", top_k)

    rows = db.execute(sql, params).fetchall()

    # ── Step 3: Filter by similarity threshold ─────────────────────
    # pgvector returns ALL rows up to LIMIT — it doesn't know our
    # threshold. We filter here in Python after getting results.
    # WHY not in SQL?
    #   We'd need: WHERE (1 - (ce.embedding <=> ...)) >= :threshold
    #   which means computing the expression TWICE in the query.
    #   Filtering the small result set (≤ top_k rows) in Python is fine.
    results = []

    for row in rows:
        similarity = float(row.similarity_score)

        if similarity < similarity_threshold:
            logger.debug(
                "Chunk filtered out | chunk_id=%s | similarity=%.3f | threshold=%.3f",
                row.chunk_id, similarity, similarity_threshold,
            )
            continue

        results.append(
            RetrievedChunk(
                chunk_id=row.chunk_id,
                content=row.content,
                filename=row.filename,
                chunk_index=row.chunk_index,
                similarity_score=round(similarity, 4),
                # WHY round to 4 decimals?
                #   similarity_score=0.7234567891 looks noise in an API response.
                #   0.7235 is clean and precise enough.
            )
        )

    logger.info(
        "Retrieval complete | query_preview='%s' | results=%d | filtered_out=%d",
        query[:50],
        len(results),
        len(rows) - len(results),
    )

    return results
