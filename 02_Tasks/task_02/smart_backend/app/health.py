# app/health.py
# Updated for task_02: adds a Redis health check on top of task_01 checks.

import logging

import redis
from fastapi import status
from fastapi.responses import JSONResponse
from openai import OpenAI

from app.auth import VALID_API_KEYS
from app.conversations import create_conversation, delete_conversation, get_conversation
from app.llm import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from app.redis_client import redis_client

logger = logging.getLogger(__name__)


def check_redis() -> str:
    """Ping Redis and report the round-trip result."""
    try:
        redis_client.ping()
        return "ok"
    except redis.RedisError as exc:
        logger.exception("Redis health check failed")
        return f"error: {exc}"


def check_api_keys() -> str:
    try:
        if not VALID_API_KEYS:
            return "error: no API keys configured"
        return "ok"
    except Exception as exc:
        logger.exception("API key health check failed")
        return f"error: {exc}"


def check_conversation_store() -> str:
    """Create → read → delete a throwaway conversation in Redis."""
    try:
        cid = create_conversation()
        get_conversation(cid)
        delete_conversation(cid)
        return "ok"
    except Exception as exc:
        logger.exception("Conversation store health check failed")
        return f"error: {exc}"


def check_llm_api() -> str:
    try:
        if not GROQ_API_KEY:
            return "error: GROQ_API_KEY is not set"
        c = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        c.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=1,
            timeout=5.0,
        )
        return "ok"
    except Exception as exc:
        logger.exception("LLM API health check failed")
        return "error: llm api unreachable"


def health_check():
    checks = {
        "redis":               check_redis(),
        "llm_api":             check_llm_api(),
        "api_keys_configured": check_api_keys(),
        "conversation_store":  check_conversation_store(),
    }

    all_healthy = all(v == "ok" for v in checks.values())

    body = {
        "status":  "healthy" if all_healthy else "degraded",
        "version": "2.0.0",
        "checks":  checks,
    }

    code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body)
