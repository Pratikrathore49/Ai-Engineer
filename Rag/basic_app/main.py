"""FastAPI app: exposes ingestion and querying over HTTP.

Think of this as the "controller" for the two RAG pipelines.
"""

import pathlib

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from config import settings
from ingest import ingest_pdf
from rag import RAGEngine
from vector_store import VectorStore

app = FastAPI(title="Basic RAG API")
store = VectorStore()
engine = RAGEngine(store)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks_indexed": store.collection.count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Upload a PDF to index it on the fly."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    path = pathlib.Path(settings.data_dir) / file.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(await file.read())

    count = ingest_pdf(store, path)
    return {"file": file.filename, "chunks_ingested": count}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty.")
    result = engine.answer(req.query)
    return QueryResponse(answer=result["answer"], sources=result["sources"])