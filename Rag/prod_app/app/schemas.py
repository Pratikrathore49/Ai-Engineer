"""Reporting models for the API contract."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    mode: str  # "normal" | "no_info" | "cached"
    faithfulness: float | None = None
    confidence: float | None = None
    latency_ms: dict


class IngestResponse(BaseModel):
    document: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    docs_ingested: list[str]