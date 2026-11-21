# app/rag/schemas.py
#
# Request and response shapes for the RAG endpoint.

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: str | None = None
    # None   → search all uploaded documents
    # a UUID → search only that specific document
    conversation_id: str | None = None
    # If provided, the question and answer are saved to this
    # conversation in Redis for follow-up questions.
    # If None, a new conversation is created automatically.
    top_k: int = Field(default=5, ge=1, le=20)


class SourceCitation(BaseModel):
    # One entry per chunk that was used to build the answer.
    filename: str
    chunk_index: int
    similarity_score: float
    # WHY return sources?
    #   This is what separates RAG from a black-box LLM.
    #   The user can verify the answer against the original document.
    #   This is called "grounded generation with citations."


class RAGResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[SourceCitation]
    chunks_used: int
    # chunks_used = len(sources) — convenience field so the client
    # doesn't have to count the array.
