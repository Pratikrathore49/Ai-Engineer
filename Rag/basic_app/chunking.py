"""Manual chunking so we understand it before trusting libraries.

Recursive splitter: try paragraph break -> sentence break -> word break,
keeping every chunk as close to chunk_size as possible without cutting mid-thought
unless the chunk is itself a single unavoidable oversized unit.
"""

import re


def _split_long(text: str, max_len: int) -> list[str]:
    """Split one text into pieces no longer than max_len.""" 
    words = text.split()
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_len:
            chunks.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks using natural boundaries.

    Input : raw text.
    Output: list of chunk strings, each approx <= chunk_size chars.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        unit = ""
        for sent in sentences:
            if len(unit) + len(sent) <= chunk_size:
                unit += " " + sent
            else:
                if unit.strip():
                    chunks.append(unit.strip())
                if len(sent) > chunk_size:
                    chunks.extend(_split_long(sent, chunk_size))
                    unit = ""
                else:
                    unit = sent
        if unit.strip():
            chunks.append(unit.strip())

    return _apply_overlap(chunks, overlap)


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Re-attach a tail of the previous chunk to the next one (boundary safety)."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        out.append(prev_tail + "\n" + chunks[i])
    return out