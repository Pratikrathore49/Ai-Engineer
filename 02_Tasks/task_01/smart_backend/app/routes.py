from fastapi import APIRouter, HTTPException
from openai import OpenAIError

from app.schemas import ChatRequest, ChatResponse
from app.services import generate_reply


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        return generate_reply(request)

    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except OpenAIError:
        raise HTTPException(
            status_code=502,
            detail="AI service request failed"
        )