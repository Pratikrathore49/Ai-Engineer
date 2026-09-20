"""Structure-aware recursive chunking (same spirit as Part 1, slightly hardened)."""

import re


def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    for para in paragraphs:
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        unit = ""
        for sent in sentences:
            if len(unit) + len(sent) <= chunk_size:
                unit += (" " + sent).strip()
            else:
                if unit.strip():
                    chunks.append(unit.strip())
                if len(sent) > chunk_size:
                    words = sent.split()
                    for i in range(0, len(words), chunk_size):
                        chunks.append(" ".join(words[i:i + chunk_size]))
                    unit = ""
                else:
                    unit = sent
        if unit.strip():
            chunks.append(unit.strip())

    if overlap > 0 and len(chunks) > 1:
        merged = [chunks[0]]
        for i, c in enumerate(chunks[1:], start=1):
            merged.append(chunks[i - 1][-overlap:] + "\n" + c)
        chunks = merged
    return chunks