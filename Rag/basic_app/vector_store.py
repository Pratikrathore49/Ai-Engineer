"""Embeddings + vector store access.

Keeps every raw vector interaction in one place so the rest of the app
never worries about the DB or model details. A "vector store module"
is exactly the swap point ChromaDB <- or another vector DB.
"""

import chromadb
from openai import OpenAI

from config import settings


class VectorStore:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.chroma = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, text: str) -> list[float]:
        """Text -> embedding vector. Same model for docs and queries."""
        resp = self.client.embeddings.create(model=settings.embedding_model, input=text)
        return resp.data[0].embedding

    def store_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """Embed every chunk and index it with its metadata."""
        vectors = [self.embed(c) for c in chunks]
        self.collection.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    def search(self, query: str, top_k: int) -> dict:
        """Embed the query and return top-k {documents, metadatas, distances}."""
        qv = self.embed(query)
        return self.collection.query(
            query_embeddings=[qv],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )