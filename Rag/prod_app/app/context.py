"""Context assembly: turn hits into a labeled, budgeted context block."""

from app.config import settings


def build_context(hits: list[dict], max_chars: int = 4000) -> tuple[str, list[dict]]:
    """Input : ranked hybrid hits.
    Output: (formatted context with [Source N] labels, source metadata list).
    Assigns source numbers in ranking order so citations are stable.
    """
    parts, sources, used = [], [], 0
    for i, hit in enumerate(hits, start=1):
        label = f"{hit['metadata'].get('source', '?')} page {hit['metadata'].get('page', '?')}"
        if used + len(hit["text"]) > max_chars:
            break
        parts.append(f"[Source {i}: {label}]\n{hit['text']}")
        sources.append({
            "source": hit["metadata"].get("source", "?"),
            "page": hit["metadata"].get("page"),
            "chunk_id": hit["chunk_id"],
            "score": hit.get("rerank_score", hit["fused_score"]),
            "rank": i,
        })
        used += len(hit["text"])
    return "\n\n".join(parts), sources