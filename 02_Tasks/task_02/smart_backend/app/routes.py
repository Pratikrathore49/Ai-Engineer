# app/routes.py

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import OpenAIError

from app.auth import verify_api_key
from app.cache import get_cache_stats
from app.conversations import get_conversation
from app.rate_limiter import rate_limit
from app.schemas import CacheStats, ChatRequest, ChatResponse
from app.services import generate_reply, stream_reply

logger = logging.getLogger(__name__)

# All routes require a valid API key
router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(request: ChatRequest, http_request: Request):
    client_ip       = http_request.client.host if http_request.client else "unknown"
    message_preview = request.message[:50]

    logger.info(
        "Incoming request | method=POST | path=/chat | ip=%s | message_preview='%s'",
        client_ip, message_preview,
    )

    try:
        return generate_reply(request)

    except ValueError as exc:
        logger.warning("Chat failed | reason=not_found | detail=%s", exc)
        raise HTTPException(status_code=404, detail="Conversation not found")

    except OpenAIError as exc:
        logger.error("LLM error | error_type=%s | detail=%s", type(exc).__name__, exc)
        raise HTTPException(status_code=502, detail="AI service request failed")


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, http_request: Request):
    client_ip       = http_request.client.host if http_request.client else "unknown"
    message_preview = request.message[:50]

    logger.info(
        "Incoming request | method=POST | path=/chat/stream | ip=%s | message_preview='%s'",
        client_ip, message_preview,
    )

    def event_generator():
        try:
            for token in stream_reply(request):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        except ValueError as exc:
            logger.warning("Stream failed | reason=not_found | detail=%s", exc)
            yield f"data: [ERROR] {exc}\n\n"

        except OpenAIError as exc:
            logger.error("LLM stream error | error_type=%s | detail=%s", type(exc).__name__, exc)
            yield "data: [ERROR] AI service request failed\n\n"

        except Exception as exc:
            logger.error("Unexpected stream error | error_type=%s | detail=%s", type(exc).__name__, exc)
            yield "data: [ERROR] streaming failed\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}")
def conversation_history(conversation_id: str):
    logger.debug("Fetching conversation | conversation_id=%s", conversation_id)

    try:
        history = get_conversation(conversation_id)
        return {"conversation_id": conversation_id, "messages": history}

    except KeyError:
        logger.warning("Conversation not found | conversation_id=%s", conversation_id)
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/cache/stats", response_model=CacheStats)
def cache_stats():
    logger.debug("Cache stats requested")
    return get_cache_stats()
