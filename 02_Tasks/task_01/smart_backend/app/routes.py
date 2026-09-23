# app/routes.py

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import OpenAIError

from app.auth import verify_api_key
from app.cache import get_cache_stats
from app.conversations import get_conversation
from app.schemas import ChatRequest, ChatResponse, CacheStats
from app.services import generate_reply, stream_reply
from app.rate_limiter import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/chat", response_model=ChatResponse,dependencies=[Depends(rate_limit)])
def chat(request: ChatRequest, http_request: Request):
    client_ip = http_request.client.host if http_request.client else "unknown"
    message_preview = request.message[:50]

    logger.info(
        "Incoming request | method=POST | path=/chat | ip=%s | message_preview='%s'",
        client_ip,
        message_preview
    )

    try:
        return generate_reply(request)

    except ValueError as e:
        logger.warning("Chat request failed | reason=not_found | detail=%s", str(e))
        raise HTTPException(status_code=404, detail="Conversation not found")

    except OpenAIError as e:
        logger.error(
            "LLM error | error_type=%s | status_code=502 | detail=%s",
            type(e).__name__,
            str(e)
        )
        raise HTTPException(status_code=502, detail="AI service request failed")

@router.post("/chat/stream")
def chat_stream(request: ChatRequest, http_request: Request):
    client_ip = http_request.client.host if http_request.client else "unknown"
    message_preview = request.message[:50]

    logger.info(
        "Incoming request | method=POST | path=/chat/stream | ip=%s | message_preview='%s'",
        client_ip,
        message_preview
    )

    def event_generator():
        try:
            for token in stream_reply(request):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        except ValueError as e:
            logger.warning("Stream failed | reason=not_found | detail=%s", str(e))
            yield f"data: [ERROR] {str(e)}\n\n"

        except OpenAIError as e:
            logger.error(
                "LLM stream error | error_type=%s | status_code=502 | detail=%s",
                type(e).__name__,
                str(e)
            )
            yield "data: [ERROR] AI service request failed\n\n"

        except Exception as e:
            logger.error("Unexpected stream error | error_type=%s | detail=%s", type(e).__name__, str(e))
            yield "data: [ERROR] streaming failed\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/conversations/{conversation_id}")
def conversation_history(conversation_id: str):
    logger.debug("Fetching conversation history | conversation_id=%s", conversation_id)

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
