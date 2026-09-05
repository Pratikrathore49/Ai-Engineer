# =============================================================================
# FILE: 03_streaming.py
# TOPIC: Streaming LLM responses — critical for good UX
#
# Streaming is how ChatGPT shows text appearing word by word.
# Without it, users stare at a blank screen for 10+ seconds.
#
# pip install openai python-dotenv fastapi uvicorn
# =============================================================================

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------------------------------------------
# PART 1: Basic streaming in the terminal
# See exactly how chunks arrive
# -----------------------------------------------------------------------------

def basic_streaming():
    """
    The simplest streaming example.
    Watch tokens arrive in real time in your terminal.
    """
    
    print("Streaming response (watch tokens appear):\n")
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    # stream=True is the only change from a normal call
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Explain what a REST API is in 100 words."}
        ],
        stream=True,
        temperature=0.7
    )
    
    full_response = ""
    
    for chunk in stream:
        # Each chunk has a delta (the new piece of text)
        delta = chunk.choices[0].delta
        
        if delta.content is not None:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"[Time to first token: {first_token_time - start_time:.2f}s]\n")
            
            token_count += 1
            print(delta.content, end="", flush=True)  # flush=True forces immediate display
            full_response += delta.content
        
        # chunk.choices[0].finish_reason tells you why the stream ended
        # "stop" = normal completion
        # "length" = hit max_tokens limit
        # "content_filter" = content was filtered
        if chunk.choices[0].finish_reason == "stop":
            total_time = time.time() - start_time
            print(f"\n\n[Stream complete in {total_time:.2f}s]")
            print(f"[Approximate tokens: ~{len(full_response.split())}]")


# -----------------------------------------------------------------------------
# PART 2: Streaming with a FastAPI endpoint
# This is how you'd actually use it in your backend.
# Because you know FastAPI, this will make immediate sense.
# -----------------------------------------------------------------------------

# Run this section with: uvicorn 03_streaming:app --reload
# Then call: GET http://localhost:8000/stream?question=What+is+Python
# You'll see text stream back in your browser or curl output

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


@app.get("/stream")
async def stream_response(question: str = "Tell me about Python"):
    """
    FastAPI endpoint that streams LLM responses.
    
    The key: use StreamingResponse with a generator function.
    This is the same pattern you'd use for streaming file downloads.
    
    Try it:
    curl -N "http://localhost:8000/stream?question=Explain+async+await"
    """
    
    def generate():
        """Generator function that yields SSE-formatted chunks."""
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Be concise."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            stream=True,
            temperature=0.7,
            max_tokens=500
        )
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                # Server-Sent Events format: "data: {content}\n\n"
                # This is the standard format for streaming APIs
                yield f"data: {delta.content}\n\n"
            
            if chunk.choices[0].finish_reason == "stop":
                # Send a special "done" event so the client knows to stop
                yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # disables nginx buffering (important in prod)
        }
    )


@app.get("/no-stream")
async def no_stream_response(question: str = "Tell me about Python"):
    """
    Same endpoint WITHOUT streaming — for comparison.
    
    Feel the difference in responsiveness between this and /stream.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return {"answer": response.choices[0].message.content}


# -----------------------------------------------------------------------------
# PART 3: Streaming with token counting
# In streaming mode, usage data comes in the LAST chunk (not all chunks)
# You need to handle this carefully
# -----------------------------------------------------------------------------

def streaming_with_usage():
    """
    How to get token usage data while streaming.
    Important for cost tracking in production.
    """
    
    print("Streaming with usage tracking:\n")
    
    # stream_options lets you get usage data in streaming mode
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is machine learning? One paragraph."}
        ],
        stream=True,
        stream_options={"include_usage": True}  # OpenAI feature to get usage in stream
    )
    
    full_text = ""
    usage_data = None
    
    for chunk in stream:
        # Regular content chunks
        if chunk.choices and chunk.choices[0].delta.content:
            text_piece = chunk.choices[0].delta.content
            full_text += text_piece
            print(text_piece, end="", flush=True)
        
        # Usage data arrives in the final chunk (after finish_reason="stop")
        if chunk.usage:
            usage_data = chunk.usage
    
    print("\n")
    
    if usage_data:
        print(f"Input tokens:  {usage_data.prompt_tokens}")
        print(f"Output tokens: {usage_data.completion_tokens}")
        print(f"Total tokens:  {usage_data.total_tokens}")


if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Basic Streaming Demo")
    print("=" * 60)
    basic_streaming()
    
    print("\n" + "=" * 60)
    print("PART 3: Streaming with Usage Tracking")
    print("=" * 60)
    streaming_with_usage()
    
    print("\n" + "=" * 60)
    print("PART 2: FastAPI Streaming Server")
    print("=" * 60)
    print("To run the FastAPI server:")
    print("  uvicorn 03_streaming:app --reload")
    print()
    print("Then test with curl:")
    print('  curl -N "http://localhost:8000/stream?question=Explain+Python+asyncio"')
    print()
    print("Or compare with non-streaming:")
    print('  curl "http://localhost:8000/no-stream?question=Explain+Python+asyncio"')
