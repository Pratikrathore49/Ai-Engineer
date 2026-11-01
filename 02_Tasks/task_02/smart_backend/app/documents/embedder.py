# app/documents/embedder.py
#
# Dual-provider embedding interface.
#
# PROVIDER SELECTION (controlled by .env):
#   EMBEDDING_PROVIDER=local   → sentence-transformers (free, runs on CPU)
#   EMBEDDING_PROVIDER=openai  → OpenAI text-embedding-3-small (paid API)
#
# WHY TWO PROVIDERS?
#   Development / cost-saving  → local model, no API key needed, no cost
#   Production / best quality  → OpenAI API, 1536-dim vectors, faster at scale
#
# VECTOR DIMENSIONS:
#   local  (all-MiniLM-L6-v2)    → 384 dimensions
#   openai (text-embedding-3-small) → 1536 dimensions
#
# ⚠️  IMPORTANT — dimensions must match your Vector() column in pgvector.
#   If you switch providers you must drop and recreate the chunk_embeddings
#   table (or re-embed everything). The column size is set by EMBEDDING_DIMENSIONS
#   below and must match what's in models.py → Vector(EMBEDDING_DIMENSIONS).
#
# HOW TO SWITCH:
#   In .env set:  EMBEDDING_PROVIDER=local   (default, free)
#             or  EMBEDDING_PROVIDER=openai  (needs OPENAI_API_KEY)
#   Restart the server — no code changes needed.

import logging
import os

logger = logging.getLogger(__name__)

# ── Read provider from environment ────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()

# ── Model names ───────────────────────────────────────────────────
LOCAL_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"

# ── Dimensions per provider ───────────────────────────────────────
# These must match the Vector(N) column in chunk_embeddings.
PROVIDER_DIMENSIONS = {
    "local":  384,
    "openai": 1536,
}

# Expose active model name and dimensions for other modules to import
EMBEDDING_MODEL      = LOCAL_MODEL_NAME if EMBEDDING_PROVIDER == "local" else OPENAI_MODEL_NAME
EMBEDDING_DIMENSIONS = PROVIDER_DIMENSIONS.get(EMBEDDING_PROVIDER, 384)


# ═══════════════════════════════════════════════════════════════════
# LOCAL PROVIDER  —  sentence-transformers (free, no API key)
# ═══════════════════════════════════════════════════════════════════

_local_model = None  # cached model instance

def _get_local_model():
    """
    Load the sentence-transformers model once and cache it.
    First call takes 2-5 seconds to download/load the model (~90MB).
    Every subsequent call returns instantly from cache.
    """
    global _local_model
    if _local_model is not None:
        return _local_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed.\n"
            "Run: pip install sentence-transformers==3.0.1"
        )

    logger.info("Loading local embedding model | model=%s", LOCAL_MODEL_NAME)
    _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
    logger.info(
        "Local embedding model loaded | dimensions=%d",
        _local_model.get_sentence_embedding_dimension(),
    )
    return _local_model


def _embed_local(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using the local sentence-transformers model.
    Runs entirely on CPU — no API calls, no cost.
    """
    model = _get_local_model()

    logger.info(
        "Local embedding | model=%s | batch_size=%d",
        LOCAL_MODEL_NAME, len(texts),
    )

    # encode() returns a numpy array of shape (len(texts), 384)
    # .tolist() converts each row to a plain Python list[float]
    embeddings = model.encode(texts, show_progress_bar=False)
    vectors = embeddings.tolist()

    logger.info(
        "Local embedding complete | vectors=%d | dimensions=%d",
        len(vectors), len(vectors[0]) if vectors else 0,
    )
    return vectors


# ═══════════════════════════════════════════════════════════════════
# OPENAI PROVIDER  —  text-embedding-3-small (paid, 1536 dims)
# ═══════════════════════════════════════════════════════════════════

_openai_client = None  # cached client instance

def _get_openai_client():
    """Return a cached OpenAI client. Reads OPENAI_API_KEY from env."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai package is not installed.\n"
            "Run: pip install openai==1.40.0"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in .env.\n"
            "Either add the key or switch to: EMBEDDING_PROVIDER=local"
        )

    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using OpenAI text-embedding-3-small.
    Batches up to 500 texts per API call (OpenAI limit is 2048).
    """
    from openai import OpenAIError

    client = _get_openai_client()
    all_vectors: list[list[float]] = []
    batch_size = 500

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]

        logger.info(
            "OpenAI embedding batch | model=%s | batch_size=%d | offset=%d",
            OPENAI_MODEL_NAME, len(batch), batch_start,
        )

        try:
            response = client.embeddings.create(
                model=OPENAI_MODEL_NAME,
                input=batch,
            )
        except OpenAIError as exc:
            logger.error(
                "OpenAI embedding failed | model=%s | error=%s",
                OPENAI_MODEL_NAME, str(exc),
            )
            raise

        batch_vectors = [item.embedding for item in response.data]
        all_vectors.extend(batch_vectors)

        logger.info(
            "OpenAI embedding batch complete | vectors=%d | dimensions=%d",
            len(batch_vectors),
            len(batch_vectors[0]) if batch_vectors else 0,
        )

    return all_vectors


# ═══════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE — the only function other modules should call
# ═══════════════════════════════════════════════════════════════════

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the configured provider.

    Provider is selected by EMBEDDING_PROVIDER in .env:
        local  → sentence-transformers (free, 384 dims)
        openai → OpenAI API (paid, 1536 dims)

    Args:
        texts: list of strings to embed

    Returns:
        List of float vectors, same order and length as input.
    """
    if not texts:
        logger.warning("embed_texts called with empty list")
        return []

    logger.debug(
        "embed_texts | provider=%s | count=%d",
        EMBEDDING_PROVIDER, len(texts),
    )

    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)

    # Default → local
    if EMBEDDING_PROVIDER != "local":
        logger.warning(
            "Unknown EMBEDDING_PROVIDER='%s', falling back to local",
            EMBEDDING_PROVIDER,
        )

    return _embed_local(texts)
