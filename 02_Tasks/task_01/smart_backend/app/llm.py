import os 

from openai import OpenAI

from app.prompts import SYSTEM_PROMPT
from app.tools import TOOLS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GROQ_MODEL = "openai/gpt-oss-120b"

client = OpenAI(api_key=GROQ_API_KEY,
base_url=GROQ_BASE_URL)

def generate_llm_response( history:list[dict]):
    """
    Send conversation history to the LLM with available tools.
    """

    messages=[
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },
        *history
    ]

    return client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )


def generate_final_llm_response(
    messages:list[dict]):

    """
    Send the tool result back to the LLM so it can generate the final answer.
    """

    return client.chat.completions.create(
        model = GROQ_MODEL,
        messages=messages,
        tool_choice="none"
    )

def stream_llm_response(history:list[dict]):

    """
    Stream LLM response using conversation history.
    """
    messages=[
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        },
        *history
    ]
    return client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        stream=True
    )
