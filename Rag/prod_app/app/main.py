"""FastAPI application entrypoint.

Routes:
  GET  /health    -> status + corpus stats
  POST /ingest    -> upload a PDF (auth required)
  POST /query     -> ask a question (auth required)
  POST /reindex   -> rebuild BM25 lexical index after ingest
  GET  /cache     -> cache stats (auth required)
"""

import pathlib
import time

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.auth import require_api_key
from app.cache import answer_cache
from app.config import settings
from app.context import build_context
from app.generation import Generation
from app.ingest import IngestionError, ingest_document, validate_upload
from app.logging_conf import get_logger, new_request_id
from app.retrieval import HybridRetriever
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse, Source
from app.vector_store import VectorStore

logger = get_logger("main")

app = FastAPI(title="Production RAG API", version="1.0.0")

store = VectorStore()
retriever = HybridRetriever(store)
generation = Generation()
_rate_buckets: dict[str, list[float]] = {}


@app.on_event("startup")
def startup() -> None:
    retriever.rebuild_lexical()
    logger.info("startup complete: %d chunks indexed", store.count())


def _check_rate_limit(key_id: str) -> None:
    now = time.time()
    last = _rate_buckets.setdefault(key_id, [])
    cutoff = now - 60
    last[:] = [t for t in last if t > cutoff]
    if len(last) >= settings.rate_limit_per_min:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    last.append(now)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    _, metas, _ = store.get_all_chunks()
    docs = sorted({m.get("source", "?") for m in metas})
    return HealthResponse(status="ok", chunks_indexed=store.count(), docs_ingested=docs)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), key_id: str = Depends(require_api_key)) -> IngestResponse:
    data = await file.read()
    try:
        validate_upload(file.filename, len(data))
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    path = pathlib.Path(settings.data_dir) / file.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    try:
        result = ingest_document(store, path)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    retriever.rebuild_lexical()  # keep lexical index fresh after ingest
    return IngestResponse(document=result.doc_name, chunks=result.chunk_count)


@app.post("/reindex")
def reindex(key_id: str = Depends(require_api_key)) -> dict:
    retriever.rebuild_lexical()
    return {"reindexed": True, "chunks": len(retriever.ids)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, key_id: str = Depends(require_api_key)) -> QueryResponse:
    request_id = new_request_id()
    _check_rate_limit(key_id)
    t0 = time.time()
    logger.info("query start request_id=%s query=%r", request_id, req.query)

    cached = answer_cache.get(req.query)
    if cached is not None:
        lat = {"total_ms": round((time.time() - t0) * 1000, 1)}
        cached["latency_ms"] = lat
        cached["mode"] = "cached"
        return QueryResponse(**cached)

    t1 = time.time()
    hits = retriever.search(req.query, top_k=req.top_k or settings.top_k)
    t2 = time.time()

    if generation.insufficient_context(hits):
        answer = "I don't have this information in the provided documents."
        lat = {"total_ms": round((time.time() - t0) * 1000, 1),
               "retrieve_ms": round((t2 - t1) * 1000, 1)}
        logger.info("circuit-breaker invoked request_id=%s hits=%d", request_id, len(hits))
        return QueryResponse(query=req.query, answer=answer, sources=[],
                             mode="no_info", faithfulness=1.0, confidence=0.0,
                             latency_ms=lat)

    t3 = time.time()
    context, sources = build_context(hits, max_chars=4000)
    answer = generation.generate(req.query, context)
    t4 = time.time()

    faithfulness = generation.check_faithfulness(answer, context)
    t5 = time.time()

    confidence = round(1.0 - min(hits[0]["distance"], 1.0), 3)
    latency = {
        "retrieve_ms": round((t2 - t1) * 1000, 1),
        "context_ms": round((t3 - t2) * 1000, 1),
        "llm_ms": round((t4 - t3) * 1000, 1),
        "guardrail_ms": round((t5 - t4) * 1000, 1),
        "total_ms": round((t5 - t0) * 1000, 1),
    }

    result = {
        "query": req.query,
        "answer": answer,
        "sources": [Source(**s) for s in sources],
        "mode": "normal",
        "faithfulness": faithfulness,
        "confidence": confidence,
        "latency_ms": latency,
    }
    logger.info("query done request_id=%s latency=%s faithfulness=%s",
                request_id, latency, faithfulness)
    answer_cache.set(req.query, result)
    return QueryResponse(**result)


@app.get("/cache")
def cache_stats(key_id: str = Depends(require_api_key)) -> dict:
    return {"size": answer_cache.size, "ttl_seconds": answer_cache.ttl}