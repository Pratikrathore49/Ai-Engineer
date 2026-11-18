
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel,Field


class DocumentBase(BaseModel):
    """Shared fields returned for a document."""
    filename: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(..., pattern=r"^(pdf|txt|md)$")
    file_size_bytes: int = Field(..., ge=0)
    status: str = Field(..., min_length=1)


class DocumentUploadResponse(DocumentBase):
    """Response returned after uploading a document."""
    document_id: UUID
    # content: str


class DocumentListItem(DocumentBase):
    """Document summary returned by GET /documents."""
    document_id: UUID
    created_at: datetime


class DocumentDetail(DocumentBase):
    """Detailed document information."""
    document_id: UUID
    created_at: datetime
    chunk_count: int = Field(..., ge=0)

# ── Search schemas ─────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request body for POST /documents/search."""
    query: str = Field(..., min_length=1, max_length=1000)
    document_id: str | None = None
    # WHY optional document_id?
    #   None  → search across ALL uploaded documents
    #   a UUID → search only chunks from that specific document
    #   This lets users either search their whole knowledge base
    #   or zoom into one specific file.
    top_k: int = Field(default=5, ge=1, le=20)
    # WHY cap at 20?
    #   More than 20 chunks in one LLM prompt blows the context window.
    #   5 is the standard default — enough context, not too noisy.

class RetrievedChunk(BaseModel):
    """One search result — a chunk with its similarity score."""
    chunk_id: str
    content: str
    filename: str
    chunk_index: int
    similarity_score: float
    # similarity_score is between 0 and 1.
    # 1.0 = identical meaning, 0.0 = completely unrelated.
    # Computed as: 1 - cosine_distance

class SearchResponse(BaseModel):
    """Full response from POST /documents/search."""
    query: str
    results: list[RetrievedChunk]
    total_results: int
