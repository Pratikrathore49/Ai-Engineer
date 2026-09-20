"""Minimal BM25 scorer (pure Python) for hybrid search.

BM25 scores document for a query by term frequency (TF) and inverse
document frequency (IDF), with length normalization. It is the lexical
arm of hybrid search: it catches exact tokens, codes, ids and names
that embedding search is weak at.

References: Robertson & Zaragoza "The Probabilistic Relevance Framework: BM25 and Beyond".
"""

import math
import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[str, int] = {}
        self.doc_len: list[int] = []
        self.docs: list[str] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def index(self, docs: list[str]) -> None:
        """Build the index over raw document strings."""
        self.docs = docs
        self.n_docs = len(docs)
        seen: set[str] = set()
        term_count: dict[str, int] = {}
        self.doc_len = []
        for doc in docs:
            tokens = _tokenize(doc)
            self.doc_len.append(len(tokens))
            seen.clear()
            for tok in tokens:
                if tok not in seen:
                    seen.add(tok)
                    term_count[tok] = term_count.get(tok, 0) + 1
        self.doc_freqs = term_count
        self.avgdl = sum(self.doc_len) / max(1, self.n_docs)

    def _idf(self, term: str) -> float:
        n = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        terms = _tokenize(query)
        doc_tokens = _tokenize(self.docs[doc_idx])
        tf: dict[str, int] = {}
        for tok in doc_tokens:
            tf[tok] = tf.get(tok, 0) + 1
        dl = self.doc_len[doc_idx]
        score = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.n_docs) if self.score(query, i) > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]