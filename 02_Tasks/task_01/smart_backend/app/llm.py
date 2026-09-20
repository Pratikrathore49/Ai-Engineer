# app/llm.py

import logging
import os

from openai import OpenAI, OpenAIError

from app.prompts import SYSTEM_PROMPT
from app.tools import TOOLS

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

def generate_llm_response(history: list[dict]):
    """
    Send conversation history to the LLM with available tools.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history
    ]

    logger.info(
        "LLM call started | model=%s | history_length=%d",
        GROQ_MODEL,
        len(history)
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        logger.info(
            "LLM call complete | model=%s | input_tokens=%d | output_tokens=%d",
            GROQ_MODEL,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        return response

    except OpenAIError as e:
        logger.error(
            "LLM API error | model=%s | error_type=%s | detail=%s",
            GROQ_MODEL,
            type(e).__name__,
            str(e)
        )
        raise

def generate_final_llm_response(messages: list[dict]):
    """
    Send the tool result back to the LLM so it can generate the final answer.
    """
    logger.info(
        "LLM final call started | model=%s | messages_count=%d",
        GROQ_MODEL,
        len(messages)
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tool_choice="none"
        )
        logger.info(
            "LLM final call complete | model=%s | input_tokens=%d | output_tokens=%d",
            GROQ_MODEL,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        return response

    except OpenAIError as e:
        logger.error(
            "LLM API error (final call) | model=%s | error_type=%s | detail=%s",
            GROQ_MODEL,
            type(e).__name__,
            str(e)
        )
        raise

def stream_llm_response(history: list[dict]):
    """
    Stream LLM response using conversation history.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history
    ]

    logger.info(
        "LLM stream started | model=%s | history_length=%d",
        GROQ_MODEL,
        len(history)
    )

    try:
        return client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            stream=True
        )
    except OpenAIError as e:
        logger.error(
            "LLM stream error | model=%s | error_type=%s | detail=%s",
            GROQ_MODEL,
            type(e).__name__,
            str(e)
        )
        raise
