from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from openai import OpenAIError

from app.cache import get_cache_stats

from app.conversations import get_conversation

from app.schemas import ChatRequest, ChatResponse, CacheStats

from app.services import generate_reply, stream_reply

from app.auth import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        return generate_reply(request)

    except ValueError as e:

        raise HTTPException(status_code=404, detail=str(e))

    except OpenAIError:

        raise HTTPException(status_code=502, detail="AI service request failed")


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):

    def event_generator():

        try:

            for token in stream_reply(request):

                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"

        except ValueError as e:

            yield f"data: [ERROR] {str(e)}\n\n"

        except OpenAIError:

            yield "data: [ERROR] AI service request failed\n\n"

        except Exception:

            yield "data: [ERROR] streaming failed\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}")
def conversation_history(conversation_id: str):

    try:

        history = get_conversation(conversation_id)

        return {"conversation_id": conversation_id, "messages": history}

    except KeyError:

        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/cache/stats", response_model=CacheStats)
def cache_stats():

    return get_cache_stats()
