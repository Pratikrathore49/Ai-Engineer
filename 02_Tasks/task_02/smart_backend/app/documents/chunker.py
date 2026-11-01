# app/documents/chunker.py
#
# Splits a large text string into overlapping word-based chunks.
#
# WHY WORD-BASED AND NOT CHARACTER-BASED?
#   Character splitting can cut in the middle of a word — "implemen" / "tation".
#   Word splitting always produces readable, complete-word chunks.
#   LLM token limits are also roughly proportional to word count, so
#   word-based chunks give more predictable sizes.
#
# WHY OVERLAP?
#   Imagine a key fact sitting at the boundary between two chunks.
#   Without overlap, that fact gets split and loses context in both chunks.
#   With overlap, the last N words of chunk K appear again at the start
#   of chunk K+1 — so the fact is fully readable in at least one chunk.
#   50-word overlap is the standard starting point.
#
# WHY SKIP CHUNKS SHORTER THAN 20 WORDS?
#   Very short chunks are usually leftover noise — a page number, a header,
#   a single line. Embedding them wastes API calls and pollutes search results
#   with low-quality matches.

import logging

logger = logging.getLogger(__name__)

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text:       the full document text
        chunk_size: number of words per chunk
        overlap:    number of words from the end of chunk K
                    that are repeated at the start of chunk K+1

    Returns:
        List of chunk strings, each between 20 and chunk_size words.

    Example with chunk_size=5, overlap=2:
        words = [A, B, C, D, E, F, G, H]
        chunk 0 → A B C D E
        chunk 1 →     D E F G H   (D and E overlap from chunk 0)
    """
    if not text or not text.strip():
        logger.warning("chunk_text called with empty text")
        return []

    # Split into words — split() handles multiple spaces/newlines automatically
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    # WHY chunk_size - overlap as the step?
    #   Each chunk starts 'step' words after the previous one.
    #   The remaining 'overlap' words are shared between the two chunks.
    #   e.g. chunk_size=500, overlap=50 → step=450
    #   chunk 0: words[0:500]
    #   chunk 1: words[450:950]   (words[450:500] overlap with chunk 0)
    #   chunk 2: words[900:1400]  (words[900:950] overlap with chunk 1)
    step = chunk_size - overlap

    # Guard: if overlap >= chunk_size the step would be 0 or negative
    # which creates an infinite loop.
    if step <= 0:
        logger.error(
            "Invalid chunking params: overlap=%d must be less than chunk_size=%d",
            overlap, chunk_size,
        )
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    chunks = []
    start = 0

    while start < total_words:
        end = start + chunk_size
        chunk_words = words[start:end]

        # Skip noise chunks (too short to be meaningful)
        if len(chunk_words) >= 20:
            chunks.append(" ".join(chunk_words))
        else:
            logger.debug(
                "Skipping short chunk | word_count=%d | start=%d",
                len(chunk_words), start,
            )

        # Move the window forward by 'step' words
        start += step

    logger.info(
        "Text chunked | total_words=%d | chunk_size=%d | overlap=%d | chunks_produced=%d",
        total_words, chunk_size, overlap, len(chunks),
    )

    return chunks
