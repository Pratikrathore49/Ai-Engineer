import os

from openai import OpenAI

from app.prompts import SYSTEM_PROMPT

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_MODEL = "openai/gpt-oss-120b"


if not GROQ_API_KEY:
    raise ValueError("Groq API key is not configured")


client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def generate_llm_response(history: list[dict[str, str]]):
    """
    Send conversation history to the LLM.
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    return client.chat.completions.create(model=GROQ_MODEL, messages=messages)


def stream_llm_response(history: list[dict[str, str]]):
    """
    Stream LLM response using conversation history.
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    return client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, stream=True
    )
