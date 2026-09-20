"""Hybrid retrieval: BM25 (lexical) + vector (semantic), fused by RRF, then optional rerank.

Call order per request:
  1. vector search top CANDIDATES
  2. bm25 rank top CANDIDATES
  3. RRF fusion -> fused ranking
  4. optional cross-encoder rerank -> final top_k
Returns enriched hits (chunk id, text, metadata, fused score).
"""

import math

from app.bm25_index import BM25
from app.config import settings
from app.reranking import rerank
from app.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.bm25 = BM25()
        self.ids: list[str] = []
        self.id_to_pos: dict[str, int] = {}

    def rebuild_lexical(self) -> None:
        ids, _, documents = self.store.get_all_chunks()
        self.ids = ids
        self.id_to_pos = {cid: i for i, cid in enumerate(ids)}
        self.bm25.index(documents)

    def _rrf_score(self, vec_rank: dict[str, int], bm25_rank: dict[str, int]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in (vec_rank, bm25_rank):
            for cid, rank in ranking.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        return fused

    def search(self, query: str, top_k: int = settings.top_k) -> list[dict]:
        """Run hybrid search and return enriched hits ranked best-first."""
        candidates = settings.retrieve_candidates

        # 1. semantic pass
        vec = self.store.search(query, candidates)
        vec_rank: dict[str, int] = {}
        for rank, cid in enumerate(vec["ids"][0]):
            vec_rank[cid] = rank

        # 2. lexical pass
        bm25_hits = self.bm25.rank(query, candidates)
        bm25_rank: dict[str, int] = {}
        for rank, (pos, _score) in enumerate(bm25_hits):
            bm25_rank[self.ids[pos]] = rank

        # 3. fusion
        fused = self._rrf_score(vec_rank, bm25_rank)
        merged_ids = sorted(fused, key=fused.get, reverse=True)

        doc_map = {cid: d for cid, d in zip(vec["ids"][0], vec["documents"][0])}
        meta_map = {cid: m for cid, m in zip(vec["ids"][0], vec["metadatas"][0])}
        dist_map = {cid: d for cid, d in zip(vec["ids"][0], vec["distances"][0])}

        hits = []
        for cid in merged_ids:
            if cid not in doc_map:
                continue
            hits.append({
                "chunk_id": cid,
                "text": doc_map[cid],
                "metadata": meta_map[cid],
                "fused_score": fused[cid],
                "distance": dist_map.get(cid, 1.0),
            })

        # 4. rerank (optional) on top fused candidates
        if hits:
            order, rerank_scores = rerank(
                query, [h["text"] for h in hits[:candidates]], [h["fused_score"] for h in hits[:candidates]]
            )
            hits = [hits[i] for i in order]
            for h, s in zip(hits, rerank_scores):
                h["rerank_score"] = s

        return hits[:top_k]