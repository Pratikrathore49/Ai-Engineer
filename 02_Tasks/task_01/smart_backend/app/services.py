import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.prompts import SYSTEM_PROMPT
from app.schemas import ChatRequest, ChatResponse


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_MODEL = "openai/gpt-oss-120b"


def generate_reply(request: ChatRequest) -> ChatResponse:

    if not GROQ_API_KEY:
        raise ValueError("Groq API key is not configured")

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
    )

    return ChatResponse(
        reply=response.choices[0].message.content,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model=response.model
    )