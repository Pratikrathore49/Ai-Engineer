# =============================================================================
# FILE: 02_tokens_and_cost.py
# TOPIC: Understanding tokens, counting them, and estimating cost
#
# pip install tiktoken openai python-dotenv
# =============================================================================

import tiktoken
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------------------------------------------
# PART 1: Tokenization without making any API calls
# tiktoken is OpenAI's tokenizer library — use it to count tokens locally
# before sending anything to the API. This is how you estimate cost upfront.
# -----------------------------------------------------------------------------


def explore_tokenization():
    """
    See exactly how different texts get tokenized.
    Run this and study the output — it changes how you write prompts.
    """

    # Use the tokenizer for gpt-4o
    enc = tiktoken.encoding_for_model("gpt-4o")

    test_cases = [
        "Hello world",
        "Python",
        "Unbelievable",
        "I love building AI applications with FastAPI",
        "supercalifragilisticexpialidocious",
        "1234567890",
        "2024-01-15",
        # Non-English text uses MORE tokens per word
        "नमस्ते दुनिया",  # Hindi: "Hello world"
        "مرحبا بالعالم",  # Arabic: "Hello world"
        "function getUserById(id: string) { return db.users.findOne(id); }",
    ]

    print(f"{'Text':<55} | {'Tokens':>6} | Token IDs")
    print("-" * 90)

    for text in test_cases:
        token_ids = enc.encode(text)
        # Decode each token individually to see how it was split
        token_pieces = [enc.decode([t]) for t in token_ids]
        print(f"{repr(text):<55} | {len(token_ids):>6} | {token_pieces}")

    print()
    print("Key insight: Non-English text = more tokens = higher cost!")
    print("English: ~0.75 words per token")
    print("Hindi/Arabic: can be 2-4 tokens per word")


# -----------------------------------------------------------------------------
# PART 2: Real-world token counting for prompts
# Always count your prompts before sending them — don't get surprised by cost
# -----------------------------------------------------------------------------


def count_prompt_tokens():
    """
    Count tokens in a complete messages array — just like the API will.
    Use this BEFORE making API calls to estimate cost.
    """

    enc = tiktoken.encoding_for_model("gpt-4o")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions about Python.",
        },
        {
            "role": "user",
            "content": "Explain list comprehensions in Python with 3 examples.",
        },
    ]

    # Count tokens for each message
    # Note: OpenAI adds ~4 tokens overhead per message for formatting
    TOKENS_PER_MESSAGE = 4
    TOKENS_FOR_REPLY_PRIMING = (
        3  # every reply is primed with <|start|>assistant<|message|>
    )

    total_tokens = 0
    for msg in messages:
        total_tokens += TOKENS_PER_MESSAGE
        for key, value in msg.items():
            total_tokens += len(enc.encode(value))
    total_tokens += TOKENS_FOR_REPLY_PRIMING

    print(f"Estimated input tokens: {total_tokens}")

    # GPT-4o pricing (as of 2025): $2.50 per 1M input tokens
    cost_per_million = 2.50
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million
    print(f"Estimated input cost:   ${estimated_cost:.6f}")
    print(f"Cost for 10,000 calls:  ${estimated_cost * 10_000:.4f}")


# -----------------------------------------------------------------------------
# PART 3: Real cost tracking from actual API responses
# Always log this in production. Token costs add up FAST.
# -----------------------------------------------------------------------------

# Pricing as of 2025 — CHECK platform.openai.com for current prices
PRICING = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,  # $ per token
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "text-embedding-3-small": {
        "input": 0.02 / 1_000_000,
        "output": 0.0,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of a single API call."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    price = PRICING[model]
    return (input_tokens * price["input"]) + (output_tokens * price["output"])


def real_cost_tracking():
    """
    Make a real API call and track its actual cost.
    This is a pattern you should use in every production app.
    """

    model = "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": "List 5 benefits of Python for AI development.",
            },
        ],
        temperature=0.7,
    )

    # Extract actual usage from response
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    cost = calculate_cost(model, input_tokens, output_tokens)

    print("=== API Call Cost Report ===")
    print(f"Model:          {model}")
    print(f"Input tokens:   {input_tokens}")
    print(f"Output tokens:  {output_tokens}")
    print(f"Total tokens:   {total_tokens}")
    print(f"Call cost:      ${cost:.6f}")
    print(f"Cost/1000 calls: ${cost * 1000:.4f}")
    print(f"Cost/10K calls:  ${cost * 10_000:.4f}")
    print()
    print("Response:")
    print(response.choices[0].message.content)


# -----------------------------------------------------------------------------
# PART 4: Context window limit in practice
# What happens when you exceed it? How to handle it gracefully?
# -----------------------------------------------------------------------------


def context_window_demo():
    """
    Shows how to check if your messages fit in the context window
    before making the API call.
    """

    MODEL_CONTEXT_LIMITS = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-3.5-turbo": 16_385,
    }

    model = "gpt-4o-mini"
    max_context = MODEL_CONTEXT_LIMITS[model]
    # Reserve tokens for the output
    max_input_tokens = max_context - 1000  # leave 1000 tokens for output

    enc = tiktoken.encoding_for_model(model)

    # Simulate a very long document (normally you'd load from a file)
    long_document = "Python is amazing. " * 5000  # ~10,000 words

    messages = [
        {"role": "system", "content": "Summarize the following document."},
        {"role": "user", "content": long_document},
    ]

    # Count tokens
    total_input_tokens = (
        sum(
            len(enc.encode(msg["content"])) + 4  # +4 per message overhead
            for msg in messages
        )
        + 3
    )

    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Max allowed:  {max_input_tokens:,}")

    if total_input_tokens > max_input_tokens:
        print(
            f"WARNING: Input too long by {total_input_tokens - max_input_tokens:,} tokens!"
        )
        print("Options:")
        print("  1. Truncate the document")
        print("  2. Split into chunks and summarize each (Map-Reduce)")
        print("  3. Use a model with larger context window")
        print("  4. Use RAG to retrieve only relevant sections")

        # Option 1: Simple truncation (not always best, but quick)
        # Cut the document so it fits
        allowed_content_tokens = (
            max_input_tokens
            - (4 * len(messages))
            - 3
            - len(enc.encode("Summarize the following document."))
        )

        doc_tokens = enc.encode(long_document)
        truncated_tokens = doc_tokens[:allowed_content_tokens]
        truncated_doc = enc.decode(truncated_tokens)

        print(f"\nTruncated to: {len(truncated_tokens):,} tokens")
        print("(In production, prefer chunking + summarization over truncation)")
    else:
        print("Input fits within context window. Safe to proceed.")


if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Tokenization Explorer")
    print("=" * 60)
    explore_tokenization()

    print("\n" + "=" * 60)
    print("PART 2: Count Prompt Tokens (No API Call)")
    print("=" * 60)
    count_prompt_tokens()

    print("\n" + "=" * 60)
    print("PART 3: Real Cost Tracking")
    print("=" * 60)
    real_cost_tracking()

    print("\n" + "=" * 60)
    print("PART 4: Context Window Management")
    print("=" * 60)
    context_window_demo()
