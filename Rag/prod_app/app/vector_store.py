"""Vector store wrapper around ChromaDB + OpenAI-compatible embeddings.

All vector/index interactions live here; swapping Chroma for another
vector DB touches only this module.
"""

import chromadb
from openai import OpenAI

from app.config import settings


class VectorStore:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.chroma = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=settings.embedding_model, input=text)
        return resp.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        all_vecs: list[list[float]] = []
        # send in batches to keep request size sane
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = self.client.embeddings.create(model=settings.embedding_model, input=batch)
            all_vecs.extend([d.embedding for d in resp.data])
        return all_vecs

    def add_chunks(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        vectors = self.embed_many(documents)
        self.collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    def search(self, query: str, top_k: int, where: dict | None = None) -> dict:
        qv = self.embed(query)
        return self.collection.query(
            query_embeddings=[qv],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

    def get_all_chunks(self) -> tuple[list[str], list[dict], list[dict]]:
        """Return (ids, metadatas, documents) for the whole collection (BM25 build)."""
        data = self.collection.get(include=["documents", "metadatas"])
        return data["ids"], data["metadatas"], data["documents"]

    def count(self) -> int:
        return self.collection.count()