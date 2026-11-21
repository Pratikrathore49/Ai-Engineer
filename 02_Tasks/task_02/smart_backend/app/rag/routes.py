# app/rag/routes.py
#
# HTTP layer for the RAG feature.
# Thin as possible — validation, logging, calling pipeline.ask().
# All business logic lives in pipeline.py, not here.

import logging

from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError
from sqlalchemy.orm import Session

from app.core.auth import verify_api_key
from app.core.db import get_db
from app.rag.pipeline import ask
from app.rag.schemas import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/ask", response_model=RAGResponse)
def rag_ask(
    request: RAGRequest,
    db: Session = Depends(get_db),
):
    """
    Full RAG pipeline — retrieve relevant document chunks,
    build a grounded prompt, generate an answer, return citations.

    Returns immediately with a "not found" message if no relevant
    chunks exist — no LLM call is made in that case.
    """
    logger.info(
        "RAG ask | question_preview='%s' | document_id=%s | top_k=%d",
        request.question[:50],
        request.document_id,
        request.top_k,
    )

    try:
        return ask(
            question=request.question,
            db=db,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
        )

    except OpenAIError as exc:
        logger.error(
            "RAG LLM error | error_type=%s | detail=%s",
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail="LLM service unavailable. Try again shortly.",
        )

    except RuntimeError as exc:
        # Raised when GROQ_API_KEY or OPENAI_API_KEY is missing
        logger.error("RAG config error | detail=%s", str(exc))
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
