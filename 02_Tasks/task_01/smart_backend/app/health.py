import logging 

from fastapi import status
from fastapi.responses import JSONResponse
from openai import OpenAI

from app.auth import VALID_API_KEYS
from app.conversations import create_conversation, get_conversation, delete_conversation
from app.llm import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

def check_api_keys():
    try:
        if not VALID_API_KEYS:
            return "error: no API keys configured"
        return "ok"
    
    except Exception as exc:
        logger.exception("API key health check failed")
        return f"error: {exc}"

def check_conversation_store():
    try:
        conversation_id = create_conversation()
        get_conversation(conversation_id)
        delete_conversation(conversation_id)
       
        return "ok"
    
    except Exception as exc:
        logger.exception("Conversation store health check failed")
        return f"error: {exc}"


def check_llm_api():
    try:
        if not GROQ_API_KEY:
            return "error: GROQ_API_KEY is not configured"
        client = OpenAI(api_key=GROQ_API_KEY, base_url='https://api.groq.com/openai/v1')

        client.chat.completions.create(
            model = GROQ_MODEL,
            messages=[
                {
                    "role":"user",
                    "content":"Say OK"
                }
            ],
        max_tokens=1,
        timeout=5.0
        )
        
        return "ok"

    except Exception as exc:
        logger.exception("LLM API health check failed")
        return f"error: llm api unreachable"

def health_check():
    checks={
        "llm_api":check_llm_api(),
        "api_keys_configured":check_api_keys(),
        "conversation_store":check_conversation_store(),
    }

    all_healthy = all(result == "ok" for result in checks.values())

    response={
        "status":"healthy" if all_healthy else "degraded",
        "version":"1.0.0",
        "checks":checks
    }

    if all_healthy:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response,
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response
    )