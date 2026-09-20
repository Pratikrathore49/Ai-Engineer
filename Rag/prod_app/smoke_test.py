"""Offline smoke test: mock the OpenAI client so no network/key is needed.

Validates the whole pipeline: ingest -> chroma store -> BM25 build ->
hybrid search -> context -> circuit breaker. Not a substitute for real eval.
"""

import os

# MUST be set before importing app modules (Settings reads env at import)
os.environ["OPENAI_API_KEY"] = "sk-dummy-for-offline-test"
os.environ["API_KEY"] = "test123"

import pathlib
import sys

from app import vector_store  # noqa: F401  (patch target module-level)

# --- deterministic fake embeddings so identical/related text is close ---
def fake_embedding(text):
    words = set(text.lower().split())
    vec = [0.0] * 64
    for w in words:
        h = hash(w) % 64
        vec[h] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def fake_embeddings_create(model, input):
    if isinstance(input, str):
        texts = [input]
    else:
        texts = input
    data = [type("D", (), {"embedding": fake_embedding(t)})() for t in texts]
    return type("R", (), {"data": data})()


vector_store.OpenAI.embeddings.create = fake_embeddings_create

from app.config import settings
from app.context import build_context
from app.generation import Generation
from app.ingest import ingest_document
from app.retrieval import HybridRetriever
from app.vector_store import VectorStore

store = VectorStore()
store.client.embeddings.create = fake_embeddings_create
retriever = HybridRetriever(store)
gen = Generation()

# --- mock LLM calls for generation (patch instance) ---
def fake_chat(self, model, messages, temperature, max_tokens):
    answer = "Employees earn 24 days of earned leave per year. [Source 1]"
    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": answer})()})()]})()


gen.client.chat.completions.create = fake_chat

# 1. ingest
total = 0
for path in sorted(pathlib.Path(settings.data_dir).glob("*.pdf")):
    r = ingest_document(store, path)
    total += r.chunk_count
    print(f"ingested {r.doc_name}: {r.chunk_count} chunks")
print("collection count:", store.collection.count())

# 2. rebuild BM25
retriever.rebuild_lexical()
print("BM25 docs:", len(retriever.ids))

# 3. hybrid search on a relevant query
hits = retriever.search("how many sick days?", top_k=3)
print("\nsick-query hits:")
for h in hits:
    print("  ", h["metadata"]["source"], round(h["fused_score"], 3),
          "| dist:", round(h["distance"], 3), "|", h["text"][:50].replace("\n", " "))

# 4. context assembly
context, sources = build_context(hits)
print("\ncontext bytes:", len(context), "sources:", len(sources))

# 5. circuit breaker on an out-of-domain query
hits2 = retriever.search("recipes for pasta", top_k=3)
print("\nout-of-domain best distance:", round(hits2[0]["distance"], 3))
breaker = gen.insufficient_context(hits2)
print("circuit-breaker triggers (no_info):", breaker)

print("\nSMOKE OK")