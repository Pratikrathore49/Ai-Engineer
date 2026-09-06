import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.prompts import SYSTEM_PROMPT
from app.schemas import ChatRequest, ChatResponse
from app.cache import get_cached_response,set_cached_response

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
        raise ValueError("Groq API key is not configured")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url= GROQ_BASE_URL
)

def generate_reply(request: ChatRequest) -> ChatResponse:
    # 1. Check cache
    cached_response = get_cached_response(request.message)
    if cached_response:
        return cached_response.model_copy(
            update={"cached":True}
        )
    # 2. No cache -> call LLM
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
    #3. Create Response
    chat_response = ChatResponse(
        reply=response.choices[0].message.content,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model=response.model,
        cached= False
    )

    # 4. Store response
    set_cached_response(
        request.message,
        chat_response
    )
    return chat_response

def stream_reply(request:ChatRequest):

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":request.message
            }
        ],
        stream=True
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content

        if token:
            yield token