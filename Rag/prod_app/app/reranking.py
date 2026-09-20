"""Optional cross-encoder reranking.

Loads a SentenceTransformers CrossEncoder lazily so the app runs even
without the optional heavy dependency (falls back to ranking by fused
score alone).
"""

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger("rerank")

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # optional dependency
        _model = CrossEncoder(settings.reranker_model)
        logger.info("loaded cross-encoder %s", settings.reranker_model)
    return _model


def rerank(query: str, documents: list[str], scores: list[float]) -> tuple[list[int], list[float]]:
    """Return (indices reordered by relevance, corresponding scores)."""
    if not settings.rerank_enabled:
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return order, [scores[i] for i in order]

    model = _get_model()
    pairs = [(query, doc) for doc in documents]
    preds = model.predict(pairs)
    order = sorted(range(len(documents)), key=lambda i: preds[i], reverse=True)
    return order, [float(preds[i]) for i in order]